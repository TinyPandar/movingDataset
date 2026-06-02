from __future__ import annotations

import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset_dark_intersection_30k_smoke" / "random"
OUT_DIR = ROOT / "box_target_examples"
OUT_SIZE = 128
DISPLAY_SCALE = 4


def load_sample() -> dict[str, str]:
    ann = DATASET / "annotations.csv"
    with ann.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # Pick a horizontal car, because the box geometry is easier to read in a compact figure.
    for row in rows:
        if row["split"] == "train" and row["rotation_deg"] in {"90", "270"}:
            return row
    return rows[0]


def scale_row(row: dict[str, str], out_size: int = OUT_SIZE) -> dict[str, float]:
    sx = out_size / float(row["image_width"])
    sy = out_size / float(row["image_height"])
    return {
        "cx": float(row["center_x"]) * sx,
        "cy": float(row["center_y"]) * sy,
        "w": float(row["car_width"]) * sx,
        "h": float(row["car_height"]) * sy,
        "rotation": float(row["rotation_deg"]),
    }


def bbox_points(g: dict[str, float]) -> list[tuple[float, float]]:
    # The generator records car_width/car_height after rotating the vehicle sprite.
    # Current dark-intersection headings are cardinal, so this is the aligned exterior box.
    x0 = g["cx"] - g["w"] / 2
    x1 = g["cx"] + g["w"] / 2
    y0 = g["cy"] - g["h"] / 2
    y1 = g["cy"] + g["h"] / 2
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def make_gaussian(g: dict[str, float], sigma: float = 3.0) -> Image.Image:
    img = Image.new("L", (OUT_SIZE, OUT_SIZE), 0)
    px = img.load()
    for y in range(OUT_SIZE):
        for x in range(OUT_SIZE):
            d2 = (x - g["cx"]) ** 2 + (y - g["cy"]) ** 2
            val = int(round(255.0 * math.exp(-d2 / (2 * sigma * sigma))))
            px[x, y] = max(px[x, y], val)
    return img


def make_targets(g: dict[str, float]) -> dict[str, Image.Image]:
    pts = bbox_points(g)
    mask = Image.new("L", (OUT_SIZE, OUT_SIZE), 0)
    d = ImageDraw.Draw(mask)
    d.polygon(pts, fill=255)

    outline = Image.new("L", (OUT_SIZE, OUT_SIZE), 0)
    d = ImageDraw.Draw(outline)
    for width in (5, 3):
        d.line([*pts, pts[0]], fill=255, width=width, joint="curve")

    center = make_gaussian(g, sigma=3.0)
    center_box = Image.blend(mask.point(lambda v: int(v * 0.38)), center, 0.55)
    # Add a faint outline so the model is still encouraged to preserve extent.
    center_box = Image.composite(Image.new("L", (OUT_SIZE, OUT_SIZE), 255), center_box, outline.point(lambda v: int(v * 0.35)))
    return {
        "box_mask": mask,
        "box_outline": outline,
        "center_box": center_box,
    }


def colorize_target(target: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    target = target.convert("L")
    base = Image.new("RGB", target.size, (6, 8, 12))
    hot = Image.new("RGB", target.size, color)
    return Image.composite(hot, base, target)


def overlay(input_img: Image.Image, target: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    img = input_img.convert("RGB")
    hot = Image.new("RGB", img.size, color)
    alpha = target.point(lambda v: int(v * 0.55))
    return Image.composite(hot, img, alpha)


def draw_input_with_box(input_img: Image.Image, g: dict[str, float]) -> Image.Image:
    img = input_img.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    pts = bbox_points(g)
    d.line([*pts, pts[0]], fill=(80, 255, 120), width=2)
    r = 2
    d.ellipse((g["cx"] - r, g["cy"] - r, g["cx"] + r, g["cy"] + r), outline=(255, 255, 255), width=1)
    return img


def label_bar(text: str, width: int, height: int = 26) -> Image.Image:
    img = Image.new("RGB", (width, height), (245, 246, 248))
    d = ImageDraw.Draw(img)
    d.text((8, 6), text, fill=(20, 25, 35))
    return img


def stack_with_label(img: Image.Image, text: str) -> Image.Image:
    display = img.resize((img.width * DISPLAY_SCALE, img.height * DISPLAY_SCALE), Image.Resampling.NEAREST)
    bar = label_bar(text, display.width, height=34)
    canvas = Image.new("RGB", (display.width, display.height + bar.height), "white")
    canvas.paste(bar, (0, 0))
    canvas.paste(display, (0, bar.height))
    return canvas


def make_single_scheme(input_img: Image.Image, target: Image.Image, name: str, color: tuple[int, int, int]) -> Image.Image:
    target_rgb = colorize_target(target, color)
    overlay_rgb = overlay(input_img, target, color)
    panels = [
        stack_with_label(input_img, "input + GT box"),
        stack_with_label(target_rgb, f"{name} target"),
        stack_with_label(overlay_rgb, "target overlay"),
    ]
    gap = 14
    w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    h = max(p.height for p in panels)
    canvas = Image.new("RGB", (w, h), "white")
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + gap
    return canvas


def make_summary(input_img: Image.Image, targets: dict[str, Image.Image], colors: dict[str, tuple[int, int, int]]) -> Image.Image:
    panels = [stack_with_label(input_img, "input + GT box")]
    for name, target in targets.items():
        panels.append(stack_with_label(colorize_target(target, colors[name]), name))
    gap = 14
    w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    h = max(p.height for p in panels)
    canvas = Image.new("RGB", (w, h), "white")
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + gap
    return canvas


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    row = load_sample()
    g = scale_row(row)
    src = Image.open(DATASET / row["image"]).convert("RGB").resize((OUT_SIZE, OUT_SIZE), Image.Resampling.BICUBIC)
    input_box = draw_input_with_box(src, g)
    targets = make_targets(g)
    colors = {
        "box_mask": (38, 199, 116),
        "box_outline": (255, 191, 0),
        "center_box": (255, 92, 92),
    }

    for name, target in targets.items():
        make_single_scheme(input_box, target, name, colors[name]).save(OUT_DIR / f"{name}_example.png")
    make_summary(input_box, targets, colors).save(OUT_DIR / "box_target_schemes_summary.png")

    meta = OUT_DIR / "sample_used.txt"
    meta.write_text(
        "\n".join(
            [
                f"image={row['image']}",
                f"center=({row['center_x']}, {row['center_y']})",
                f"car_width={row['car_width']}",
                f"car_height={row['car_height']}",
                f"rotation_deg={row['rotation_deg']}",
                f"output_size={OUT_SIZE}",
            ]
        ),
        encoding="utf-8",
    )
    print(OUT_DIR)


if __name__ == "__main__":
    main()
