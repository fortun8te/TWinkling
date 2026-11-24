#!/usr/bin/env python3
"""Write the image fixtures of the vision tests to native/tests/golden/vision/fixtures/.

The fixtures are small, deterministic images that cover the cases of the image path: one token, sizes that
are not a multiple of 40 (padding), every kind of transparency the reference processor composites onto its
checkerboard (RGBA, grey + alpha, palette), a tall strip, a grid above 32 patches per side (the position
table is upsampled), a large image for the long-sequence path, a JPEG and a GIF (decoder tolerance), WebP files
(lossy, lossless with transparency, animated) and a rendered line of text for the end-to-end check. tools/vision_ref.py turns them into goldens.

Needs pillow and numpy (tools/.venv-vision). Rerunning writes the same pixels; commit the files.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "native" / "tests" / "golden" / "vision" / "fixtures"


def smooth(height, width, seed):
    """A smooth RGB pattern (compresses well): three phase-shifted 2-D sinusoids plus a diagonal ramp."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    rng = np.random.default_rng(seed)
    out = np.empty((height, width, 3))
    for c in range(3):
        fy, fx, ph = rng.uniform(0.004, 0.03), rng.uniform(0.004, 0.03), rng.uniform(0, 6.28)
        out[..., c] = 0.5 + 0.35 * np.sin(fy * y + fx * x + ph) + 0.15 * ((x + y) / (height + width) - 0.5)
    return np.clip(np.rint(out * 255), 0, 255).astype(np.uint8)


def noise(height, width, channels, seed):
    return np.random.default_rng(seed).integers(0, 256, (height, width, channels), dtype=np.uint8)


def text_image(text, width=320, height=80):
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=48)
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(((width - box[2] + box[0]) // 2 - box[0], (height - box[3] + box[1]) // 2 - box[1]), text,
              fill=(0, 0, 0), font=font)
    return image


def build():
    images = {}
    images["solid_red_40x40.png"] = Image.new("RGB", (40, 40), (255, 0, 0))
    images["solid_blue_40x40.png"] = Image.new("RGB", (40, 40), (0, 0, 255))
    images["grad_100x60.png"] = Image.fromarray(smooth(60, 100, 1))  # 3 x 2 tokens, padded right and bottom
    rgba = noise(64, 64, 4, 2)
    rgba[..., 3] = np.linspace(0, 255, 64).astype(np.uint8)[None, :]  # alpha ramp across the checkerboard
    images["rgba_64x64.png"] = Image.fromarray(rgba, "RGBA")
    la = noise(48, 48, 2, 3)
    la[:24, :, 1] = 255
    images["la_48x48.png"] = Image.fromarray(la, "LA")
    palette = Image.fromarray(noise(44, 56, 1, 4)[..., 0] % 8, "P")
    palette.putpalette([v for i in range(8) for v in (i * 36, 255 - i * 30, (i * 97) % 256)])
    palette.info["transparency"] = bytes([0, 64, 128, 255, 255, 255, 255, 255])
    images["palette_trns_56x44.png"] = palette
    images["noise_200x120.png"] = Image.fromarray(noise(120, 200, 3, 5))  # 5 x 3 = 15 tokens
    images["strip_40x1240.png"] = Image.fromarray(smooth(1240, 40, 6))  # 1 column, 31 rows
    images["medium_720x680.png"] = Image.fromarray(smooth(680, 720, 7))  # 18 x 17 tokens: grid 36 x 34 > 32
    # 43 x 43 = 1849 tokens; 4-pixel blocks keep the file small
    images["big_1720x1720.png"] = Image.fromarray(np.kron(smooth(430, 430, 8), np.ones((4, 4, 1), np.uint8)))
    images["photo_160x120.jpg"] = Image.fromarray(smooth(120, 160, 9))
    gif = Image.fromarray(noise(40, 80, 1, 10)[..., 0] % 16, "P")
    gif.putpalette([v for i in range(16) for v in (i * 17, (i * 53) % 256, 255 - i * 17)])
    images["palette_80x40.gif"] = gif
    images["text_inkling42_320x80.png"] = text_image("INKLING 42")
    # WebP: lossy, lossless with transparency, and an animation (the model sees its first frame)
    images["lossy_160x120.webp"] = Image.fromarray(smooth(120, 160, 11))
    alpha = noise(56, 72, 4, 12)
    alpha[..., 3] = np.linspace(0, 255, 56).astype(np.uint8)[:, None]
    images["lossless_alpha_72x56.webp"] = Image.fromarray(alpha, "RGBA")
    images["animated_80x80.webp"] = [Image.fromarray(smooth(80, 80, 13)), Image.fromarray(smooth(80, 80, 14))]
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=FIXTURES, help=f"output folder (default: {FIXTURES})")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, image in build().items():
        path = args.out / name
        if isinstance(image, list):  # animation
            image[0].save(path, save_all=True, append_images=image[1:], duration=200, lossless=True)
            image = image[0]
        elif name.endswith(".webp"):
            image.save(path, lossless=image.mode == "RGBA", quality=85, method=6)
        elif name.endswith(".jpg"):
            image.save(path, quality=90, subsampling=0)
        elif name.endswith(".gif"):
            image.save(path)
        else:
            image.save(path, optimize=True, **({"transparency": image.info["transparency"]}
                                                if "transparency" in image.info else {}))
        print(f"{name}: {image.size[0]}x{image.size[1]} {image.mode}, {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
