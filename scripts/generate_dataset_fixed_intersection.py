import argparse
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from generate_dataset import (
    ROAD_PALETTE,
    add_preview_marker,
    clear_pngs,
    draw_crosswalk,
    make_vehicle_sprite,
    place_vehicle_in_scene,
    validate_annotations,
)


FIXED_BACKGROUND_STYLE = "fixed_intersection"
FIXED_CAR_COLOR = "blue"
FIXED_CAR_SCALE = 0.85


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a fixed-background single-intersection vehicle localization dataset."
    )
    parser.add_argument("--output", type=Path, default=Path("dataset_fixed_intersection"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--safety-margin", type=int, default=32)
    parser.add_argument("--debug-preview-count", type=int, default=10)
    return parser.parse_args()


def make_fixed_intersection_background(image_size: int) -> tuple[Image.Image, list[dict[str, object]]]:
    image = Image.new("RGB", (image_size, image_size), ROAD_PALETTE["sidewalk"])
    draw = ImageDraw.Draw(image)

    grass_w = int(round(image_size * 0.045))
    draw.rectangle((0, 0, grass_w, image_size), fill=ROAD_PALETTE["grass"])
    draw.rectangle((image_size - grass_w, 0, image_size, image_size), fill=ROAD_PALETTE["grass"])
    draw.rectangle((0, 0, image_size, grass_w), fill=ROAD_PALETTE["grass"])
    draw.rectangle((0, image_size - grass_w, image_size, image_size), fill=ROAD_PALETTE["grass"])

    road_w = int(round(image_size * 0.42))
    cx = image_size // 2
    cy = image_size // 2
    x0 = cx - road_w // 2
    x1 = cx + road_w // 2
    y0 = cy - road_w // 2
    y1 = cy + road_w // 2

    draw.rectangle((x0, 0, x1, image_size), fill=ROAD_PALETTE["asphalt_dark"])
    draw.rectangle((0, y0, image_size, y1), fill=ROAD_PALETTE["asphalt_dark"])

    curb_w = 8
    draw.rectangle((x0 - curb_w, 0, x0, image_size), fill=ROAD_PALETTE["curb"])
    draw.rectangle((x1, 0, x1 + curb_w, image_size), fill=ROAD_PALETTE["curb"])
    draw.rectangle((0, y0 - curb_w, image_size, y0), fill=ROAD_PALETTE["curb"])
    draw.rectangle((0, y1, image_size, y1 + curb_w), fill=ROAD_PALETTE["curb"])

    side_line_w = 4
    for offset in (x0 + 18, x1 - 22):
        draw.rectangle((offset, 0, offset + side_line_w, image_size), fill=ROAD_PALETTE["line_white"])
    for offset in (y0 + 18, y1 - 22):
        draw.rectangle((0, offset, image_size, offset + side_line_w), fill=ROAD_PALETTE["line_white"])

    dash_len = 42
    dash_gap = 30
    current_y = 18
    while current_y < image_size:
        draw.rectangle((cx - 5, current_y, cx + 5, current_y + dash_len), fill=ROAD_PALETTE["line_yellow"])
        current_y += dash_len + dash_gap

    current_x = 18
    while current_x < image_size:
        draw.rectangle((current_x, cy - 5, current_x + dash_len, cy + 5), fill=ROAD_PALETTE["line_yellow"])
        current_x += dash_len + dash_gap

    draw_crosswalk(draw, x0 + 12, y0 - 54, road_w - 24, 36, horizontal=True)
    draw_crosswalk(draw, x0 + 12, y1 + 18, road_w - 24, 36, horizontal=True)
    draw_crosswalk(draw, x0 - 54, y0 + 12, 36, road_w - 24, horizontal=False)
    draw_crosswalk(draw, x1 + 18, y0 + 12, 36, road_w - 24, horizontal=False)

    buffer = 48
    regions = [
        {"bbox": (x0 + 34, 24, x1 - 34, y0 - buffer), "headings": (0, 180), "jitter_deg": 0},
        {"bbox": (x0 + 34, y1 + buffer, x1 - 34, image_size - 24), "headings": (0, 180), "jitter_deg": 0},
        {"bbox": (24, y0 + 34, x0 - buffer, y1 - 34), "headings": (90, 270), "jitter_deg": 0},
        {"bbox": (x1 + buffer, y0 + 34, image_size - 24, y1 - 34), "headings": (90, 270), "jitter_deg": 0},
    ]
    return image, regions


def build_sample(image_size: int, safety_margin: int, rng: random.Random) -> tuple[Image.Image, dict[str, object]]:
    background, regions = make_fixed_intersection_background(image_size)
    sprite, vehicle_mask = make_vehicle_sprite(scale=FIXED_CAR_SCALE, color_name=FIXED_CAR_COLOR)
    rotated_sprite, placement, rotation_deg = place_vehicle_in_scene(
        image_size=image_size,
        safety_margin=safety_margin,
        sprite=sprite,
        vehicle_mask=vehicle_mask,
        regions=regions,
        rng=rng,
    )

    canvas = background.convert("RGBA")
    canvas.alpha_composite(rotated_sprite, dest=(int(placement["paste_x"]), int(placement["paste_y"])))
    final_image = canvas.convert("RGB")
    metadata = {
        "center_x": placement["center_x"],
        "center_y": placement["center_y"],
        "car_width": placement["car_width"],
        "car_height": placement["car_height"],
        "rotation_deg": rotation_deg,
        "scale": FIXED_CAR_SCALE,
        "background_style": FIXED_BACKGROUND_STYLE,
        "car_color": FIXED_CAR_COLOR,
    }
    return final_image, metadata


def generate_dataset(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    output_root = args.output.resolve()
    train_dir = output_root / "images" / "train"
    val_dir = output_root / "images" / "val"
    preview_dir = output_root / "debug_preview"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    clear_pngs(train_dir)
    clear_pngs(val_dir)
    clear_pngs(preview_dir)

    train_count = int(round(args.count * args.train_ratio))
    train_count = max(0, min(args.count, train_count))
    debug_count = max(0, min(args.count, args.debug_preview_count))
    preview_rng = random.Random(args.seed + 999)
    debug_indices = set(preview_rng.sample(range(args.count), k=debug_count)) if debug_count else set()

    records: list[dict[str, object]] = []
    for index in range(args.count):
        split = "train" if index < train_count else "val"
        image, metadata = build_sample(image_size=args.image_size, safety_margin=args.safety_margin, rng=rng)
        filename = f"sample_{index:05d}.png"
        relative_path = Path("images") / split / filename
        target_dir = train_dir if split == "train" else val_dir
        image.save(target_dir / filename)

        record = {
            "image": relative_path.as_posix(),
            "split": split,
            "image_width": args.image_size,
            "image_height": args.image_size,
            **metadata,
        }
        records.append(record)

        if index in debug_indices:
            preview = add_preview_marker(image, float(metadata["center_x"]), float(metadata["center_y"]))
            preview.save(preview_dir / filename)

    validate_annotations(records, args.image_size, train_count, args.count)
    pd.DataFrame.from_records(records).to_csv(output_root / "annotations.csv", index=False)


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be a positive integer.")
    if args.image_size < 128:
        raise ValueError("--image-size must be at least 128.")
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1.")
    if args.safety_margin < 0:
        raise ValueError("--safety-margin must be non-negative.")
    generate_dataset(args)


if __name__ == "__main__":
    main()
