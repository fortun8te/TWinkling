#!/usr/bin/env python3
"""Requantize Inkling-2-Mini's MXFP8 expert layers (20-23) to NVFP4 for 16 GB GPUs (ingine RTX 5080 port).

Writes OUT as a model folder that ingine loads like the original: every original file is symlinked, except
  * experts-nvfp4.safetensors: the requantized layers' w13/w2 weight, scale, amax and outer_scale_denominator
  * model.safetensors.index.json: those tensors mapped to the new file, everything else to the original shard
  * hf_quant_config.json: the layers moved from the mxfp8 to the nvfp4 parameter group
NVFP4 encoding (what ingine's kernels and tools/inkling_ref.py decode): value = E2M1(code) * E4M3(scale per 16
consecutive values of a row) * amax / denominator (per expert and matrix), low nibble = even column. The
denominator is the checkpoint's own (448 * 6 = 2688 in layers 0-19). The per-block scale is chosen by search:
the E4M3 scales around blockmax / 6 and blockmax / 4 (plus one ulp either side), keeping the one with the least
squared error over the block's 16 decoded values (the source values are the exact MXFP8 decodes).

    python3 requant_nvfp4.py --model-dir models/Inkling-2-Mini --out models/Inkling-2-Mini-nvfp4 [--layers 20-23]
Needs torch (CPU) and safetensors.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
MID = (E2M1[1:] + E2M1[:-1]) / 2  # decision points between magnitudes


def ranges(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def mxfp8_decode(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """[R, C] e4m3 codes, [R, C/32] E8M0 -> fp32 [R, C]"""
    r, c = codes.shape
    v = codes.float().view(r, c // 32, 32) * torch.exp2(scales.float() - 127)[..., None]
    return v.view(r, c)


def e4m3_round(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0, 448).to(torch.float8_e4m3fn).float()


def e4m3_step(s: torch.Tensor, up: bool) -> torch.Tensor:
    """neighbouring E4M3 value (positive, finite)"""
    b = s.to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int16)
    b = (b + (1 if up else -1)).clamp(1, 0x7E)
    return b.to(torch.uint8).view(torch.float8_e4m3fn).float()


def quantize_codes(y: torch.Tensor) -> torch.Tensor:
    """y (values in units of the block scale) -> E2M1 codes 0..15 (sign bit 8), nearest magnitude (ties down)"""
    mag = torch.bucketize(y.abs(), MID, right=False)  # 0..7
    return mag + 8 * (y < 0).to(mag.dtype)


def nvfp4_encode(x: torch.Tensor, den: float):
    """x fp32 [R, C] (one expert matrix) -> (packed u8 [R, C/2], e4m3 scales [R, C/16], amax fp32, mse)"""
    r, c = x.shape
    amax = x.abs().max().clamp_min(1e-30)
    f = amax / den  # factor
    xb = x.view(r, c // 16, 16)
    bmax = xb.abs().amax(-1)
    base6 = e4m3_round(bmax / 6 / f)
    base4 = e4m3_round(bmax / 4 / f)
    cands = [base6, e4m3_step(base6, True), e4m3_step(base6, False), base4, e4m3_step(base4, True)]
    best_err = None
    best_s = None
    best_codes = None
    lut = E2M1
    for s in cands:
        s = torch.where(s > 0, s, torch.full_like(s, 2.0**-9))  # smallest subnormal E4M3 for all-zero blocks
        y = xb / (s * f)[..., None]
        codes = quantize_codes(y)
        mag = lut[codes & 7]
        val = torch.where(codes >= 8, -mag, mag) * (s * f)[..., None]
        err = ((val - xb) ** 2).sum(-1)
        if best_err is None:
            best_err, best_s, best_codes = err, s, codes
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best_s = torch.where(take, s, best_s)
            best_codes = torch.where(take[..., None], codes, best_codes)
    codes = best_codes.view(r, c).to(torch.uint8)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    mse = float(best_err.sum()) / x.numel()
    return packed.contiguous(), best_s.to(torch.float8_e4m3fn).contiguous(), float(amax), mse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", default="20-23")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    src = args.model_dir.resolve()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads((src / "model.safetensors.index.json").read_text())
    quant = json.loads((src / "hf_quant_config.json").read_text())
    layers = ranges(args.layers)
    wm = index["weight_map"]
    den = None
    with safe_open(str(src / wm["model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator"]), "pt") as f:
        den = float(f.get_tensor("model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator").reshape(-1)[0])
    print(f"NVFP4 outer scale denominator (layer 0): {den}")
    tensors: dict[str, torch.Tensor] = {}
    report = {}
    for l in layers:
        p = f"model.llm.layers.{l}.mlp.experts."
        with safe_open(str(src / wm[p + "w13_weight"]), "pt") as f:
            if f.get_tensor(p + "w13_weight").dtype != torch.float8_e4m3fn:
                raise SystemExit(f"layer {l} is not MXFP8")
            for w in ("w13", "w2"):
                t0 = time.time()
                codes = f.get_tensor(p + f"{w}_weight")
                scales = f.get_tensor(p + f"{w}_weight_scale")
                E, R, C = codes.shape
                packed = torch.empty(E, R, C // 2, dtype=torch.uint8)
                bs = torch.empty(E, R, C // 16, dtype=torch.float8_e4m3fn)
                amax = torch.empty(E, dtype=torch.float32)
                rel = []
                for e in range(E):
                    x = mxfp8_decode(codes[e], scales[e])
                    packed[e], bs[e], amax[e], mse = nvfp4_encode(x, den)
                    rel.append(mse / float((x * x).mean()))
                tensors[p + f"{w}_weight"] = packed
                tensors[p + f"{w}_weight_scale"] = bs
                tensors[p + f"{w}_weight_amax"] = amax
                tensors[p + f"{w}_weight_outer_scale_denominator"] = torch.tensor([den], dtype=torch.float32)
                r = sum(rel) / len(rel)
                report[f"layer{l}.{w}"] = {"relative_mse": r, "relative_rms": r**0.5}
                print(f"layer {l} {w}: {E}x{R}x{C} relative RMS error {r ** 0.5:.4f} ({time.time() - t0:.0f} s)", flush=True)
    shard = "experts-nvfp4.safetensors"
    save_file(tensors, str(out / shard), metadata={"format": "pt", "source": "ingine tools/requant_nvfp4.py"})
    # index: replaced/added tensors -> new shard
    new_wm = dict(wm)
    for l in layers:
        p = f"model.llm.layers.{l}.mlp.experts."
        for k in list(new_wm):
            if k.startswith(p):
                del new_wm[k]
    for k in tensors:
        new_wm[k] = shard
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": index.get("metadata", {}), "weight_map": new_wm}, indent=2))
    # quant config: move w13/w2 names of the layers from mxfp8 to nvfp4
    names = {f"model.llm.layers.{l}.mlp.experts.{w}_weight" for l in layers for w in ("w13", "w2")}
    groups = quant["model"]["parameter_groups"]
    nv = next(g for g in groups if g["compute_dtype"] == "nvfp4")
    for g in groups:
        if g["compute_dtype"] == "mxfp8":
            g["tensors"] = [t for t in g["tensors"] if t not in names]
    nv["tensors"] = sorted(set(nv["tensors"]) | names)
    quant["model"]["parameter_groups"] = [g for g in groups if g["tensors"]]
    (out / "hf_quant_config.json").write_text(json.dumps(quant, indent=2))
    # everything else: symlinks
    for item in src.iterdir():
        if item.name in ("model.safetensors.index.json", "hf_quant_config.json") or item.name.startswith("."):
            continue
        dst = out / item.name
        if not dst.exists() and not dst.is_symlink():
            os.symlink(item, dst)
    (out / "requant-report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
