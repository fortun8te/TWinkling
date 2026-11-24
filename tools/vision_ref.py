#!/usr/bin/env python3
"""Reference outputs of Inkling-2-Mini's image path, for ingine's vision tests.

Runs the Hugging Face transformers implementation (the pinned commit of tools/requirements.txt) of the
image processor, the vision tower and its projector on the fixtures of tools/vision_fixtures.py and writes
one golden file per fixture to native/tests/golden/vision/<fixture>.safetensors:

  grid             i32 [2]            rows, cols of 40x40 tiles = language-model tokens
  rows             i32 [R]            the image-token rows kept in image_embeds (all of them up to 64 tokens)
  image_embeds     f32 [R, 2304]      projector output with FP32 arithmetic
  bf16_error       f32 [2]            relative RMS error and cosine of the same output with the model in BF16,
                                      as transformers serves it, against image_embeds: the tolerance of the tests
  rgb              u8  [H, W, 3]      the processor's RGB pixels (alpha composited), before padding (not for
                                      the two large fixtures)
  pixel_values     f32 [N, 40, 40, 3] the processor's output                      } FULL fixtures only
  embeddings_in    f32 [S, 1024]      class token + patches + position embeddings }
  layer_<i>        f32 [S, 1024]      residual stream after encoder layer 0, 11, 22 }

Only the vision tower is built (318M parameters), on the CPU: no GPU and no language model are needed.

--check-spec also runs an independent NumPy (FP64) implementation written from the description of the
model in native/KERNELS.md and compares it with transformers; it guards the description, not the code.

Needs its own small environment (CPU torch; the GPU environment of tools/requirements.txt works too):

  uv venv --python 3.12 tools/.venv-vision
  uv pip install --python tools/.venv-vision/bin/python -r tools/requirements-vision.txt
  tools/.venv-vision/bin/python tools/vision_fixtures.py && tools/.venv-vision/bin/python tools/vision_ref.py --check-spec
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "native" / "tests" / "golden" / "vision"
TILE, PATCH, MERGE, GRID, HIDDEN, HEADS, LAYERS = 40, 20, 2, 32, 1024, 16, 23
KEEP_LAYERS = (0, 11, 22)
MAX_ROWS = 64  # image_embeds rows kept per fixture
# fixtures that also keep the processor output and intermediate states
FULL = ("solid_red_40x40", "grad_100x60", "rgba_64x64", "noise_200x120")
MAX_RGB_BYTES = 512 * 1024  # larger fixtures keep no rgb: the decoder is covered by the small ones

# checkpoint name (after "model.visual.") -> transformers Inkling2VisionModel name
RENAMES = (("backbone.embeddings.position_embedding", "embeddings.position_embedding.weight"),
           ("backbone.embeddings.", "embeddings."), ("backbone.ln_pre.", "pre_layernorm."),
           ("backbone.layers.", "layers."), (".ln_1.", ".layernorm_before."), (".ln_2.", ".layernorm_after."),
           (".ls_1.gamma", ".lambda_1"), (".ls_2.gamma", ".lambda_2"), (".mlp.c_fc.", ".mlp.fc1."),
           (".mlp.c_proj.", ".mlp.fc2."), ("proj_in.", "merger.linear_1."), ("proj_out.", "merger.linear_2."))


def checkpoint_tensors(model_dir):
    """{checkpoint name without "model.visual.": BF16 tensor} of the vision tower and projector."""
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    names = sorted(name for name in index if name.startswith("model.visual."))
    tensors, handles = {}, {}
    for name in names:
        file = index[name]
        if file not in handles:
            handles[file] = safe_open(str(model_dir / file), framework="pt", device="cpu")
        tensors[name.removeprefix("model.visual.")] = handles[file].get_tensor(name)
    return tensors


def load_tower(model_dir, tensors):
    from transformers.models.inkling2 import Inkling2VisionConfig, Inkling2VisionModel
    config = json.loads((model_dir / "config.json").read_text())["vision_config"]
    config.pop("model_type", None)
    model = Inkling2VisionModel(Inkling2VisionConfig(**config))
    state = {}
    for name, tensor in tensors.items():
        for old, new in RENAMES:
            if old in name:
                name = name.replace(old, new)
        state[name] = tensor
    model.load_state_dict(state, strict=True)  # every checkpoint tensor has a home and none is missing
    return model.eval()


def reference(model, processor, image):
    """Processor + tower for one PIL image, in the model's current dtype. Returns a dict of FP32 arrays."""
    inputs = processor(images=[image], return_tensors="pt")
    pixels, grid = inputs["pixel_values"], inputs["image_grid_thw"]
    out, hooks = {}, []

    def keep(name):
        def hook(_module, _args, output):
            out[name] = output.detach().float().numpy()
        return hook

    hooks.append(model.embeddings.register_forward_hook(keep("embeddings_in")))
    for i, layer in enumerate(model.layers):
        hooks.append(layer.register_forward_hook(keep(f"layer_{i}")))
    with torch.no_grad():
        result = model(pixel_values=pixels, image_grid_thw=grid)
    for hook in hooks:
        hook.remove()
    out["image_embeds"] = result.pooler_output.float().numpy()
    out["pixel_values"] = pixels.float().numpy()
    out["grid"] = grid[0, 1:].numpy().astype(np.int32)
    return out


def processor_rgb(pixel_values, grid, height, width):
    """The processor's unpadded RGB bytes, recovered from its normalised tiles."""
    rows, cols = int(grid[0]), int(grid[1])
    canvas = pixel_values.reshape(rows, cols, TILE, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(rows * TILE, cols * TILE, 3)
    rgb = np.rint((canvas[:height, :width].astype(np.float64) * 0.5 + 0.5) * 255)
    assert rgb.min() >= 0 and rgb.max() <= 255
    return rgb.astype(np.uint8)


# ---- independent NumPy implementation of the description (FP64) ------------------------------------------

def spec_rgb(image):
    """Decoded pixels -> RGB bytes: transparency is composited onto an 8-pixel checkerboard of 204 / 153."""
    if not image.has_transparency_data:
        return np.asarray(image.convert("RGB"))
    rgba = np.asarray(image.convert("RGBA")).astype(np.int64)
    height, width = rgba.shape[:2]
    y, x = np.mgrid[0:height, 0:width]
    background = np.where((y // 8 + x // 8) % 2 == 0, 204, 153)[..., None]
    alpha = rgba[..., 3:4]
    return ((rgba[..., :3] * alpha + background * (255 - alpha) + 127) // 255).astype(np.uint8)


def spec_erf(x):
    return torch.special.erf(torch.from_numpy(x)).numpy()  # elementwise special function only


def spec_gelu(x):
    return 0.5 * x * (1.0 + spec_erf(x / math.sqrt(2.0)))


def spec_layernorm(x, weight, bias, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def spec_forward(rgb, w):
    """rgb u8 [H, W, 3], w: {checkpoint name: FP64 array}. Returns (embeddings_in, {layer: state}, embeds)."""
    height, width = rgb.shape[:2]
    rows, cols = math.ceil(height / TILE), math.ceil(width / TILE)
    canvas = np.full((rows * TILE, cols * TILE, 3), -1.0)
    canvas[:height, :width] = rgb
    canvas = (canvas / 255.0 - 0.5) / 0.5
    patches, patch_row, patch_col = [], [], []
    for tile_row in range(rows):          # one language-model token per tile, in raster order
        for tile_col in range(cols):
            for in_row in range(MERGE):   # its four patches: (0,0), (0,1), (1,0), (1,1)
                for in_col in range(MERGE):
                    y0, x0 = tile_row * TILE + in_row * PATCH, tile_col * TILE + in_col * PATCH
                    patches.append(canvas[y0:y0 + PATCH, x0:x0 + PATCH].reshape(-1))  # (y, x, channel)
                    patch_row.append(tile_row * MERGE + in_row)
                    patch_col.append(tile_col * MERGE + in_col)
    patch_row, patch_col = np.array(patch_row), np.array(patch_col)
    x = np.stack(patches) @ w["backbone.embeddings.patch_embedding.weight"].T

    table = w["backbone.embeddings.position_embedding"].reshape(GRID, GRID, HIDDEN)

    def taps(index, size):  # bilinear, half-pixel centres, edge clamp
        src = (index + 0.5) * GRID / size - 0.5
        low = np.floor(src)
        frac = src - low
        return (np.clip(low, 0, GRID - 1).astype(int), np.clip(low + 1, 0, GRID - 1).astype(int), 1.0 - frac, frac)

    r0, r1, wr0, wr1 = taps(patch_row, rows * MERGE)
    c0, c1, wc0, wc1 = taps(patch_col, cols * MERGE)
    x = x + ((wr0 * wc0)[:, None] * table[r0, c0] + (wr0 * wc1)[:, None] * table[r0, c1]
             + (wr1 * wc0)[:, None] * table[r1, c0] + (wr1 * wc1)[:, None] * table[r1, c1])
    cls = w["backbone.embeddings.class_embedding"] + w["backbone.embeddings.cls_position_embedding"]
    x = np.concatenate([cls[None], x])
    embeddings_in = x
    x = spec_layernorm(x, w["backbone.ln_pre.weight"], w["backbone.ln_pre.bias"])

    # 2-D rotary positions: patches at (row + 1, col + 1), the class token at (0, 0)
    head = HIDDEN // HEADS
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, head // 2, 2) / (head // 2)))        # 16 frequencies
    pos_row = np.concatenate([[0], patch_row + 1])[:, None] * inv_freq
    pos_col = np.concatenate([[0], patch_col + 1])[:, None] * inv_freq
    angle = np.repeat(np.concatenate([pos_col, pos_row], axis=-1), 2, axis=-1)       # pairs 0-15 col, 16-31 row
    cos, sin = np.cos(angle)[:, None, :], np.sin(angle)[:, None, :]

    def rotate(t):  # interleaved pairs (a, b) -> (a cos - b sin, b cos + a sin)
        a, b = t[..., 0::2], t[..., 1::2]
        turned = np.stack([-b, a], axis=-1).reshape(t.shape)
        return t * cos + turned * sin

    states = {}
    seq = x.shape[0]
    for i in range(LAYERS):
        p = f"backbone.layers.{i}."
        y = spec_layernorm(x, w[p + "ln_1.weight"], w[p + "ln_1.bias"])
        q = (y @ w[p + "self_attn.q_proj.weight"].T + w[p + "self_attn.q_proj.bias"]).reshape(seq, HEADS, head)
        k = (y @ w[p + "self_attn.k_proj.weight"].T + w[p + "self_attn.k_proj.bias"]).reshape(seq, HEADS, head)
        v = (y @ w[p + "self_attn.v_proj.weight"].T + w[p + "self_attn.v_proj.bias"]).reshape(seq, HEADS, head)
        q, k = rotate(q), rotate(k)
        scores = np.einsum("qhd,khd->hqk", q, k) / math.sqrt(head)
        scores = np.exp(scores - scores.max(-1, keepdims=True))
        attn = np.einsum("hqk,khd->qhd", scores / scores.sum(-1, keepdims=True), v).reshape(seq, HIDDEN)
        attn = attn @ w[p + "self_attn.out_proj.weight"].T + w[p + "self_attn.out_proj.bias"]
        x = x + w[p + "ls_1.gamma"] * attn
        y = spec_layernorm(x, w[p + "ln_2.weight"], w[p + "ln_2.bias"])
        y = spec_gelu(y @ w[p + "mlp.c_fc.weight"].T + w[p + "mlp.c_fc.bias"])
        x = x + w[p + "ls_2.gamma"] * (y @ w[p + "mlp.c_proj.weight"].T + w[p + "mlp.c_proj.bias"])
        states[i] = x
    merged = x[1:].reshape(rows * cols, MERGE * MERGE * HIDDEN)  # drop the class token; 4 patches per token
    embeds = spec_gelu(merged @ w["proj_in.weight"].T) @ w["proj_out.weight"].T
    return embeddings_in, states, embeds


def rel_rms(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean()))


def cosine(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / math.sqrt((a @ a) * (b @ b)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "Inkling-2-Mini")
    parser.add_argument("--fixtures", type=Path, default=GOLDEN / "fixtures")
    parser.add_argument("--out", type=Path, default=GOLDEN)
    parser.add_argument("--only", nargs="*", help="fixture file names (default: all)")
    parser.add_argument("--check-spec", action="store_true", help="compare the NumPy description with transformers")
    parser.add_argument("--spec-max-tokens", type=int, default=400, help="largest fixture --check-spec runs")
    parser.add_argument("--no-write", action="store_true", help="do not write golden files")
    args = parser.parse_args()

    from transformers.models.inkling2 import Inkling2ImageProcessor
    torch.manual_seed(0)
    tensors = checkpoint_tensors(args.model_dir)
    print(f"vision tensors: {len(tensors)}, {sum(t.numel() for t in tensors.values()) / 1e6:.1f}M parameters")
    processor = Inkling2ImageProcessor.from_pretrained(args.model_dir)
    model32 = load_tower(args.model_dir, tensors).float()
    model16 = load_tower(args.model_dir, tensors).to(torch.bfloat16)
    w64 = {name: t.double().numpy() for name, t in tensors.items()} if args.check_spec else None

    failures = 0
    summary = {}
    names = args.only or sorted(p.name for p in args.fixtures.iterdir() if p.suffix in (".png", ".jpg", ".gif", ".webp"))
    for name in names:
        image = Image.open(args.fixtures / name)
        width, height = image.size
        ref = reference(model32, processor, image)
        ref16 = reference(model16, processor, image)
        rows, cols = int(ref["grid"][0]), int(ref["grid"][1])
        tokens = rows * cols
        rgb = processor_rgb(ref["pixel_values"], ref["grid"], height, width)
        bf16_err = rel_rms(ref16["image_embeds"], ref["image_embeds"])
        bf16_cos = cosine(ref16["image_embeds"], ref["image_embeds"])
        line = f"{name}: {width}x{height} -> {rows}x{cols} = {tokens} tokens; bf16 vs fp32 rel-rms {bf16_err:.2e} cos {bf16_cos:.6f}"
        summary[name] = {"width": width, "height": height, "rows": rows, "cols": cols,
                         "bf16_rel_rms": bf16_err, "bf16_cosine": bf16_cos}

        if args.check_spec and tokens <= args.spec_max_tokens:
            spec_pixels = spec_rgb(image)
            pixel_ok = np.array_equal(spec_pixels, rgb)
            emb_in, states, embeds = spec_forward(spec_pixels, w64)
            errs = {"embeddings_in": rel_rms(emb_in, ref["embeddings_in"]),
                    "layer_0": rel_rms(states[0], ref["layer_0"]),
                    f"layer_{LAYERS - 1}": rel_rms(states[LAYERS - 1], ref[f"layer_{LAYERS - 1}"]),
                    "image_embeds": rel_rms(embeds, ref["image_embeds"])}
            worst = max(errs.values())
            ok = pixel_ok and worst <= 1e-4
            failures += not ok
            line += f"\n    spec: pixels {'equal' if pixel_ok else 'DIFFER'}; rel-rms " + \
                    ", ".join(f"{k} {v:.1e}" for k, v in errs.items()) + ("" if ok else "  <-- MISMATCH")
        print(line, flush=True)

        if not args.no_write:
            stem = Path(name).stem
            keep = np.arange(tokens, dtype=np.int32) if tokens <= MAX_ROWS else \
                np.unique(np.linspace(0, tokens - 1, MAX_ROWS).round().astype(np.int32))
            golden = {"grid": ref["grid"], "rows": keep,
                      "image_embeds": ref["image_embeds"][keep].astype(np.float32),
                      "bf16_error": np.array([bf16_err, bf16_cos], np.float32)}
            if rgb.size <= MAX_RGB_BYTES:
                golden["rgb"] = rgb
            if stem in FULL:
                golden["pixel_values"] = ref["pixel_values"].astype(np.float32)
                golden["embeddings_in"] = ref["embeddings_in"].astype(np.float32)
                for i in KEEP_LAYERS:
                    golden[f"layer_{i}"] = ref[f"layer_{i}"].astype(np.float32)
            args.out.mkdir(parents=True, exist_ok=True)
            save_file(golden, str(args.out / (stem + ".safetensors")))
    if not args.no_write:
        path = args.out / "summary.json"
        if args.only and path.exists():  # keep the fixtures not run this time
            summary = dict(sorted({**json.loads(path.read_text()), **summary}.items()))
        path.write_text(json.dumps(summary, indent=1) + "\n")
    if failures:
        sys.exit(f"{failures} fixture(s) differ from the description")


if __name__ == "__main__":
    main()
