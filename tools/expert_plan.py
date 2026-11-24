#!/usr/bin/env python3
"""Routed-expert placement plans for ingine's RTX 5080 port (RuntimeOptions::expert_plan, --expert-plan).

A plan orders each layer's 128 experts and splits them into `hot` (VRAM), `keep - hot` (pinned host memory,
read by the GPU over PCIe) and `128 - keep` pruned (never routed; the top-8 weights renormalise over the rest).

    expert_plan.py --host-layers 20-23 -o plan.json            whole layers' experts in host memory
    expert_plan.py --stats calib.bin --model-dir M --vram-gib 10.5 [--prune 16] [--prune-layers 0-23] -o plan.json

With --stats (ingine-kl --calib output: per visit and expert {pairs, sum of weights, sum of w*||out||}):
  * pruning removes, per layer, the --prune experts with the lowest REAP saliency (mean router weight times
    expert output norm over the tokens routed to it, summed over the layer's visits: router-weighted, not a
    routing count, since the bias-balanced router keeps counts flat);
  * VRAM placement fills --vram-gib with the kept experts most often routed per byte (summed over visits:
    looped layers 6-17 run three times per token), the rest go to host memory.
Expert sizes come from the checkpoint (NVFP4 4.27 MiB, MXFP8 7.83 MiB per expert).
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

LAYERS, EXPERTS = 24, 128
VISITS = list(range(6)) + [l for _ in range(3) for l in range(6, 18)] + list(range(18, 24))
NVFP4_BYTES = 2304 * 1152 + 2304 * 144 + 2304 * 576 + 2304 * 72
MXFP8_BYTES = 2304 * 2304 + 2304 * 72 + 2304 * 1152 + 2304 * 36


def ranges(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if not part:
            continue
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def read_stats(path: Path):
    raw = Path(path).read_bytes()
    assert raw[:4] == b"IKS1", "not an ingine-kl --calib file"
    visits, experts, tokens = struct.unpack_from("<iiq", raw, 4)
    vals = struct.unpack_from(f"<{visits * experts * 3}f", raw, 20)
    stats = [[vals[(v * experts + e) * 3:(v * experts + e) * 3 + 3] for e in range(experts)] for v in range(visits)]
    return stats, tokens


def expert_bytes(model_dir: Path | None) -> list[int]:
    fmt = ["nvfp4"] * 20 + ["mxfp8"] * 4
    if model_dir is not None:
        q = json.loads((Path(model_dir) / "hf_quant_config.json").read_text())
        for g in q["model"]["parameter_groups"]:
            for t in g["tensors"]:
                if t.endswith("mlp.experts.w13_weight"):
                    fmt[int(t.split(".")[3])] = g["compute_dtype"]
    return [NVFP4_BYTES if f == "nvfp4" else MXFP8_BYTES for f in fmt]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--host-layers", default="", help="layers whose experts all live in host memory, e.g. 20-23")
    ap.add_argument("--stats", type=Path)
    ap.add_argument("--model-dir", type=Path, help="checkpoint (expert formats) for --vram-gib")
    ap.add_argument("--vram-gib", type=float, help="VRAM for experts (GiB); default: all kept experts")
    ap.add_argument("--prune", type=int, default=0, help="experts removed per layer (lowest REAP saliency)")
    ap.add_argument("--prune-layers", default="0-23")
    args = ap.parse_args()

    plan = {"layers": []}
    if args.stats is None:
        for l in ranges(args.host_layers):
            plan["layers"].append({"layer": l, "hot": 0})
        Path(args.out).write_text(json.dumps(plan) + "\n")
        print(f"{args.out}: experts of layers {args.host_layers} in host memory")
        return

    stats, tokens = read_stats(args.stats)
    count = [[0.0] * EXPERTS for _ in range(LAYERS)]
    wsum = [[0.0] * EXPERTS for _ in range(LAYERS)]
    sal = [[0.0] * EXPERTS for _ in range(LAYERS)]
    for v, l in enumerate(VISITS):
        for e in range(EXPERTS):
            c, w, s = stats[v][e]
            count[l][e] += c
            wsum[l][e] += w
            sal[l][e] += s
    prune_layers = set(ranges(args.prune_layers))
    sizes = expert_bytes(args.model_dir)
    host_layers = set(ranges(args.host_layers))
    kept: list[list[int]] = []
    for l in range(LAYERS):
        # REAP: mean over the tokens of the layer of w * ||out|| (sum over routed tokens / tokens)
        order = sorted(range(EXPERTS), key=lambda e: -sal[l][e])
        n = args.prune if l in prune_layers else 0
        kept.append(order[:EXPERTS - n])
    # VRAM: kept experts by routing count per byte, globally
    budget = None if args.vram_gib is None else args.vram_gib * 2**30
    cand = sorted(((count[l][e] / sizes[l], l, e) for l in range(LAYERS) if l not in host_layers for e in kept[l]),
                  reverse=True)
    hot = {l: set() for l in range(LAYERS)}
    used = 0
    for _, l, e in cand:
        if budget is not None and used + sizes[l] > budget:
            continue
        hot[l].add(e)
        used += sizes[l]
    total_pairs = sum(sum(count[l]) for l in range(LAYERS))
    cold_pairs = 0.0
    for l in range(LAYERS):
        k = kept[l]
        hot_l = [e for e in k if e in hot[l]]
        cold_l = sorted((e for e in k if e not in hot[l]), key=lambda e: -count[l][e])
        pruned = [e for e in range(EXPERTS) if e not in set(k)]
        order = hot_l + cold_l + pruned
        plan["layers"].append({"layer": l, "order": order, "hot": len(hot_l), "keep": len(k)})
        cold_pairs += sum(count[l][e] for e in cold_l)
    pruned_mass = sum(sum(wsum[l][e] for e in range(EXPERTS) if e not in set(kept[l])) for l in range(LAYERS))
    all_mass = sum(sum(wsum[l]) for l in range(LAYERS))
    plan["summary"] = {
        "calibration_tokens": tokens,
        "vram_expert_gib": used / 2**30,
        "host_expert_gib": sum(sizes[l] * (len(kept[l]) - len(hot[l])) for l in range(LAYERS)) / 2**30,
        "pruned_per_layer": args.prune,
        "pruned_router_weight_share": pruned_mass / all_mass if all_mass else 0.0,
        "cold_pair_share": cold_pairs / total_pairs if total_pairs else 0.0,
        "cold_experts_per_token": 8 * len(VISITS) * cold_pairs / total_pairs if total_pairs else 0.0,
    }
    Path(args.out).write_text(json.dumps(plan) + "\n")
    print(json.dumps(plan["summary"], indent=1))


if __name__ == "__main__":
    main()
