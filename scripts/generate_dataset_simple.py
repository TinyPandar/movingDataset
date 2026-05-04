import argparse
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from generate_dataset import (
    ROAD_PALETTE,
    add_preview_marker,
    add_region,
    apply_render_variation,
    clear_pngs,
    draw_asphalt_texture,
    draw_crosswalk,
    make_vehicle_sprite,
    place_vehicle_in_scene,
    validate_annotations,
)


SIMPLE_BACKGROUND_STYLES = (
    "straight_lane",
    "parking_lot",
    "intersection",
    "curbside",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the simple single-road / single-intersection vehicle localization dataset."
    )
    parser.add_argument("--output", type=Path, default=Path("dataset_simple"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--safety-margin", type=int, default=32)
    parser.add_argument("--debug-preview-count", type=int, default=10)
    return parser.parse_args()


def make_simple_background(image_size: int, rng: random.Random) -> tuple[Image.Image, str, list[dict[str, object]]]:
    style = rng.choice(SIMPLE_BACKGROUND_STYLES)
    base_color = ROAD_PALETTE["sidewalk"] if style in {"intersection", "straight_lane"} else ROAD_PALETTE["asphalt"]
    image = Image.new("RGB", (image_size, image_size), base_color)
    draw = ImageDraw.Draw(image)
    regions: list[dict[str, object]] = []

    if style == "parking_lot":
        draw_asphalt_texture(image, rng)

    if style == "straight_lane":
        grass_w = rng.randint(26, 44)
        draw.rectangle((0, 0, grass_w, image_size), fill=ROAD_PALETTE["grass"])
        draw.rectangle((image_size - grass_w, 0, image_size, image_size), fill=ROAD_PALETTE["grass"])
        lane_x = image_size // 2
        lane_width = rng.randint(image_size // 3, image_size // 2)
        left = lane_x - lane_width // 2
        right = lane_x + lane_width // 2
        draw.rectangle((left, 0, right, image_size), fill=ROAD_PALETTE["asphalt_dark"])
        road_patch = image.crop((left, 0, right, image_size))
        draw_asphalt_texture(road_patch, rng)
        image.paste(road_patch, (left, 0))
        draw.rectangle((left - 8, 0, left, image_size), fill=ROAD_PALETTE["curb"])
        draw.rectangle((right, 0, right + 8, image_size), fill=ROAD_PALETTE["curb"])
        draw.rectangle((left + 18, 0, left + 22, image_size), fill=ROAD_PALETTE["line_white"])
        draw.rectangle((right - 22, 0, right - 18, image_size), fill=ROAD_PALETTE["line_white"])

        current_y = 18
        while current_y < image_size:
            draw.rectangle((lane_x - 5, current_y, lane_x + 5, current_y + 42), fill=ROAD_PALETTE["line_yellow"])
            current_y += 74

        crosswalk_bounds = None
        if rng.random() < 0.5:
            cw_y = rng.randint(image_size // 5, image_size - image_size // 5)
            draw_crosswalk(draw, left + 10, cw_y, lane_width - 20, 42, horizontal=True)
            crosswalk_bounds = (cw_y - 42, cw_y + 86)

        road_bbox = (left + 34, 24, right - 34, image_size - 24)
        if crosswalk_bounds:
            add_region(regions, (road_bbox[0], road_bbox[1], road_bbox[2], crosswalk_bounds[0]), (0, 180), 12)
            add_region(regions, (road_bbox[0], crosswalk_bounds[1], road_bbox[2], road_bbox[3]), (0, 180), 12)
        else:
            add_region(regions, road_bbox, (0, 180), 12)

    elif style == "parking_lot":
        draw.rectangle((0, 0, image_size, image_size), fill=(84, 86, 89))
        draw_asphalt_texture(image, rng)
        cols = rng.randint(4, 6)
        slot_w = image_size // cols
        slot_h = rng.randint(118, 146)
        for row_y in (48, image_size - slot_h - 48):
            for col in range(cols):
                x0 = col * slot_w + 18
                x1 = (col + 1) * slot_w - 18
                draw.rectangle((x0, row_y, x1, row_y + slot_h), outline=ROAD_PALETTE["line_white"], width=3)
                add_region(regions, (x0 + 4, row_y + 4, x1 - 4, row_y + slot_h - 4), (0, 180), 0)

        aisle_y0 = image_size // 2 - 64
        aisle_y1 = image_size // 2 + 64
        draw.rectangle((0, aisle_y0, image_size, aisle_y1), fill=ROAD_PALETTE["asphalt_dark"])
        for x in range(32, image_size, 84):
            draw.rectangle((x, image_size // 2 - 4, x + 40, image_size // 2 + 4), fill=ROAD_PALETTE["line_yellow"])
        if not regions:
            add_region(regions, (24, aisle_y0 + 10, image_size - 24, aisle_y1 - 10), (90, 270), 6)

    elif style == "intersection":
        road_w = rng.randint(image_size // 3, image_size // 2)
        x0 = image_size // 2 - road_w // 2
        x1 = image_size // 2 + road_w // 2
        y0 = x0
        y1 = x1
        draw.rectangle((x0, 0, x1, image_size), fill=ROAD_PALETTE["asphalt_dark"])
        draw.rectangle((0, y0, image_size, y1), fill=ROAD_PALETTE["asphalt_dark"])
        vertical_patch = image.crop((x0, 0, x1, image_size))
        horizontal_patch = image.crop((0, y0, image_size, y1))
        draw_asphalt_texture(vertical_patch, rng)
        draw_asphalt_texture(horizontal_patch, rng)
        image.paste(vertical_patch, (x0, 0))
        image.paste(horizontal_patch, (0, y0))
        draw.rectangle((x0 - 8, 0, x0, image_size), fill=ROAD_PALETTE["curb"])
        draw.rectangle((x1, 0, x1 + 8, image_size), fill=ROAD_PALETTE["curb"])
        draw.rectangle((0, y0 - 8, image_size, y0), fill=ROAD_PALETTE["curb"])
        draw.rectangle((0, y1, image_size, y1 + 8), fill=ROAD_PALETTE["curb"])

        for offset in (x0 + 18, x1 - 22):
            draw.rectangle((offset, 0, offset + 4, image_size), fill=ROAD_PALETTE["line_white"])
        for offset in (y0 + 18, y1 - 22):
            draw.rectangle((0, offset, image_size, offset + 4), fill=ROAD_PALETTE["line_white"])

        for y in range(30, image_size, 76):
            draw.rectangle((image_size // 2 - 5, y, image_size // 2 + 5, y + 36), fill=ROAD_PALETTE["line_yellow"])
        for x in range(30, image_size, 76):
            draw.rectangle((x, image_size // 2 - 5, x + 36, image_size // 2 + 5), fill=ROAD_PALETTE["line_yellow"])

        draw_crosswalk(draw, x0 + 12, y0 - 54, road_w - 24, 36, horizontal=True)
        draw_crosswalk(draw, x0 + 12, y1 + 18, road_w - 24, 36, horizontal=True)
        draw_crosswalk(draw, x0 - 54, y0 + 12, 36, road_w - 24, horizontal=False)
        draw_crosswalk(draw, x1 + 18, y0 + 12, 36, road_w - 24, horizontal=False)

        add_region(regions, (x0 + 34, 24, x1 - 34, y0 - 96), (0, 180), 10)
        add_region(regions, (x0 + 34, y1 + 96, x1 - 34, image_size - 24), (0, 180), 10)
        add_region(regions, (24, y0 + 34, x0 - 96, y1 - 34), (90, 270), 10)
        add_region(regions, (x1 + 96, y0 + 34, image_size - 24, y1 - 34), (90, 270), 10)

    else:
        split_x = rng.randint(image_size // 4, image_size // 3)
        draw.rectangle((0, 0, split_x, image_size), fill=ROAD_PALETTE["sidewalk"])
        draw.rectangle((split_x, 0, image_size, image_size), fill=ROAD_PALETTE["asphalt_dark"])
        road_patch = image.crop((split_x, 0, image_size, image_size))
        draw_asphalt_texture(road_patch, rng)
        image.paste(road_patch, (split_x, 0))
        draw.rectangle((0, 0, 22, image_size), fill=ROAD_PALETTE["grass"])
        draw.rectangle((split_x - 10, 0, split_x, image_size), fill=ROAD_PALETTE["curb"])
        for y in range(24, image_size, 82):
            draw.rectangle((split_x + 24, y, split_x + 34, y + 50), fill=ROAD_PALETTE["line_white"])
            draw.rectangle((image_size - 32, y, image_size - 22, y + 50), fill=ROAD_PALETTE["line_white"])
        for y in range(46, image_size - 120, 132):
            draw.rectangle((split_x + 42, y, split_x + 152, y + 74), outline=ROAD_PALETTE["line_white"], width=3)
        add_region(regions, (split_x + 58, 24, image_size - 52, image_size - 24), (0, 180), 10)

    return image, style, regions


def build_sample(image_size: int, safety_margin: int, rng: random.Random) -> tuple[Image.Image, dict[str, object]]:
    for _ in range(120):
        background, background_style, regions = make_simple_background(image_size, rng)
        if not regions:
            continue
        color_name = rng.choice(
            ("red", "blue", "green", "yellow", "white", "black", "silver", "orange")
        )
        scale = round(rng.uniform(0.7, 1.3), 3)
        sprite, vehicle_mask = make_vehicle_sprite(scale=scale, color_name=color_name)
        try:
            rotated_sprite, placement, rotation_deg = place_vehicle_in_scene(
                image_size=image_size,
                safety_margin=safety_margin,
                sprite=sprite,
                vehicle_mask=vehicle_mask,
                regions=regions,
                rng=rng,
            )
        except RuntimeError:
            continue

        canvas = background.convert("RGBA")
        canvas.alpha_composite(rotated_sprite, dest=(int(placement["paste_x"]), int(placement["paste_y"])))
        final_image = apply_render_variation(canvas.convert("RGB"), rng)
        metadata = {
            "center_x": placement["center_x"],
            "center_y": placement["center_y"],
            "car_width": placement["car_width"],
            "car_height": placement["car_height"],
            "rotation_deg": rotation_deg,
            "scale": scale,
            "background_style": background_style,
            "car_color": color_name,
        }
        return final_image, metadata

    raise RuntimeError("Failed to build a valid simple sample after multiple retries.")


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
