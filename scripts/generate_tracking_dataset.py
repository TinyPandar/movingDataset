import argparse
import csv
import json
import math
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from generate_dataset import (
    CAR_COLORS,
    ROAD_PALETTE,
    add_preview_marker,
    apply_render_variation,
    draw_asphalt_texture,
    draw_crosswalk,
    make_vehicle_sprite,
    rotate_vehicle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a single-vehicle tracking dataset on a simple cross intersection."
    )
    parser.add_argument("--output", type=Path, default=Path("tracking_dataset"))
    parser.add_argument("--sequence-count", type=int, default=24)
    parser.add_argument("--frames-per-sequence", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-preview-count", type=int, default=6)
    return parser.parse_args()


def lerp_point(p0: tuple[float, float], p1: tuple[float, float], t: float) -> tuple[float, float]:
    return (
        p0[0] + (p1[0] - p0[0]) * t,
        p0[1] + (p1[1] - p0[1]) * t,
    )


def quadratic_bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    u = 1.0 - t
    return (
        u * u * p0[0] + 2.0 * u * t * p1[0] + t * t * p2[0],
        u * u * p0[1] + 2.0 * u * t * p1[1] + t * t * p2[1],
    )


def quadratic_tangent(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    return (
        2.0 * (1.0 - t) * (p1[0] - p0[0]) + 2.0 * t * (p2[0] - p1[0]),
        2.0 * (1.0 - t) * (p1[1] - p0[1]) + 2.0 * t * (p2[1] - p1[1]),
    )


def heading_from_vector(dx: float, dy: float) -> int:
    return int(round((math.degrees(math.atan2(dy, dx)) + 90.0) % 360.0))


def build_intersection_scene(
    image_size: int,
    rng: random.Random,
) -> tuple[Image.Image, dict[str, object], list[dict[str, object]]]:
    image = Image.new("RGB", (image_size, image_size), ROAD_PALETTE["sidewalk"])
    draw = ImageDraw.Draw(image)

    grass_w = rng.randint(22, 36)
    draw.rectangle((0, 0, grass_w, image_size), fill=ROAD_PALETTE["grass"])
    draw.rectangle((image_size - grass_w, 0, image_size, image_size), fill=ROAD_PALETTE["grass"])
    draw.rectangle((0, 0, image_size, grass_w), fill=ROAD_PALETTE["grass"])
    draw.rectangle((0, image_size - grass_w, image_size, image_size), fill=ROAD_PALETTE["grass"])

    road_w = rng.randint(image_size // 3, image_size // 2)
    cx = image_size / 2.0
    cy = image_size / 2.0
    x0 = int(round(cx - road_w / 2.0))
    x1 = int(round(cx + road_w / 2.0))
    y0 = int(round(cy - road_w / 2.0))
    y1 = int(round(cy + road_w / 2.0))

    draw.rectangle((x0, 0, x1, image_size), fill=ROAD_PALETTE["asphalt_dark"])
    draw.rectangle((0, y0, image_size, y1), fill=ROAD_PALETTE["asphalt_dark"])

    vertical_patch = image.crop((x0, 0, x1, image_size))
    horizontal_patch = image.crop((0, y0, image_size, y1))
    draw_asphalt_texture(vertical_patch, rng)
    draw_asphalt_texture(horizontal_patch, rng)
    image.paste(vertical_patch, (x0, 0))
    image.paste(horizontal_patch, (0, y0))

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

    lane_offset = max(24, int(round(road_w * 0.18)))

    dash_len = rng.randint(34, 46)
    dash_gap = rng.randint(24, 32)
    current_y = 18
    while current_y < image_size:
        draw.rectangle((int(cx) - 5, current_y, int(cx) + 5, current_y + dash_len), fill=ROAD_PALETTE["line_yellow"])
        current_y += dash_len + dash_gap

    current_x = 18
    while current_x < image_size:
        draw.rectangle((current_x, int(cy) - 5, current_x + dash_len, int(cy) + 5), fill=ROAD_PALETTE["line_yellow"])
        current_x += dash_len + dash_gap

    draw_crosswalk(draw, x0 + 12, y0 - 54, road_w - 24, 36, horizontal=True)
    draw_crosswalk(draw, x0 + 12, y1 + 18, road_w - 24, 36, horizontal=True)
    draw_crosswalk(draw, x0 - 54, y0 + 12, 36, road_w - 24, horizontal=False)
    draw_crosswalk(draw, x1 + 18, y0 + 12, 36, road_w - 24, horizontal=False)

    margin = 46.0
    north_in = (cx + lane_offset, margin)
    south_out = (cx + lane_offset, image_size - margin)
    south_in = (cx - lane_offset, image_size - margin)
    north_out = (cx - lane_offset, margin)
    west_in = (margin, cy - lane_offset)
    east_out = (image_size - margin, cy - lane_offset)
    east_in = (image_size - margin, cy + lane_offset)
    west_out = (margin, cy + lane_offset)

    routes = [
        {"route_id": "north_to_south", "trajectory_type": "straight", "start": north_in, "end": south_out},
        {"route_id": "south_to_north", "trajectory_type": "straight", "start": south_in, "end": north_out},
        {"route_id": "west_to_east", "trajectory_type": "straight", "start": west_in, "end": east_out},
        {"route_id": "east_to_west", "trajectory_type": "straight", "start": east_in, "end": west_out},
        {
            "route_id": "north_to_east",
            "trajectory_type": "turn",
            "start": north_in,
            "control": (north_in[0], east_out[1]),
            "end": east_out,
        },
        {
            "route_id": "east_to_south",
            "trajectory_type": "turn",
            "start": east_in,
            "control": (south_out[0], east_in[1]),
            "end": south_out,
        },
        {
            "route_id": "south_to_west",
            "trajectory_type": "turn",
            "start": south_in,
            "control": (south_in[0], west_out[1]),
            "end": west_out,
        },
        {
            "route_id": "west_to_north",
            "trajectory_type": "turn",
            "start": west_in,
            "control": (north_out[0], west_in[1]),
            "end": north_out,
        },
    ]

    scene_meta = {
        "background_style": "simple_intersection_track",
        "road_width": road_w,
        "center_x": cx,
        "center_y": cy,
        "lane_offset": lane_offset,
    }
    return image, scene_meta, routes


def sample_route_points(route: dict[str, object], frame_count: int) -> tuple[list[tuple[float, float]], list[int]]:
    points: list[tuple[float, float]] = []
    headings: list[int] = []
    if route["trajectory_type"] == "straight":
        start = tuple(route["start"])
        end = tuple(route["end"])
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        heading = heading_from_vector(dx, dy)
        for idx in range(frame_count):
            t = idx / max(1, frame_count - 1)
            points.append(lerp_point(start, end, t))
            headings.append(heading)
        return points, headings

    start = tuple(route["start"])
    control = tuple(route["control"])
    end = tuple(route["end"])
    for idx in range(frame_count):
        t = idx / max(1, frame_count - 1)
        points.append(quadratic_bezier(start, control, end, t))
        dx, dy = quadratic_tangent(start, control, end, t)
        headings.append(heading_from_vector(dx, dy))
    return points, headings


def clamp_center(center: float, radius: float, low: float, high: float) -> float:
    return max(low + radius, min(high - radius, center))


def build_tracking_sequence(
    image_size: int,
    frame_count: int,
    rng: random.Random,
) -> tuple[list[Image.Image], dict[str, object], list[dict[str, object]]]:
    for _ in range(120):
        background, scene_meta, routes = build_intersection_scene(image_size, rng)
        route = rng.choice(routes)

        color_name = rng.choice(tuple(CAR_COLORS.keys()))
        scale = round(rng.uniform(0.72, 1.0), 3)
        sprite, vehicle_mask = make_vehicle_sprite(scale=scale, color_name=color_name)
        points, headings = sample_route_points(route, frame_count)

        frames: list[Image.Image] = []
        keyframes: list[dict[str, object]] = []
        success = True
        for frame_idx, ((cx, cy), base_heading) in enumerate(zip(points, headings)):
            rotation_deg = (base_heading + rng.randint(-3, 3)) % 360
            rotated_sprite, vehicle_bbox = rotate_vehicle(sprite, vehicle_mask, rotation_deg)
            vehicle_width = vehicle_bbox[2] - vehicle_bbox[0]
            vehicle_height = vehicle_bbox[3] - vehicle_bbox[1]
            cx = clamp_center(cx, vehicle_width / 2.0, 24.0, image_size - 24.0)
            cy = clamp_center(cy, vehicle_height / 2.0, 24.0, image_size - 24.0)

            bbox_x = cx - vehicle_width / 2.0
            bbox_y = cy - vehicle_height / 2.0
            paste_x = int(round(bbox_x - vehicle_bbox[0]))
            paste_y = int(round(bbox_y - vehicle_bbox[1]))
            if (
                paste_x < 0
                or paste_y < 0
                or paste_x + rotated_sprite.size[0] > image_size
                or paste_y + rotated_sprite.size[1] > image_size
            ):
                success = False
                break

            canvas = background.convert("RGBA")
            canvas.alpha_composite(rotated_sprite, dest=(paste_x, paste_y))
            final_frame = apply_render_variation(canvas.convert("RGB"), rng)
            frames.append(final_frame)
            keyframes.append(
                {
                    "frame_idx": frame_idx,
                    "center_x": round(cx, 2),
                    "center_y": round(cy, 2),
                    "rotation_deg": rotation_deg,
                    "car_width": vehicle_width,
                    "car_height": vehicle_height,
                }
            )

        if success and len(frames) == frame_count:
            meta = {
                "background_style": scene_meta["background_style"],
                "route_id": route["route_id"],
                "trajectory_type": route["trajectory_type"],
                "car_color": color_name,
                "scale": scale,
                "image_width": image_size,
                "image_height": image_size,
            }
            return frames, meta, keyframes

    raise RuntimeError("Failed to build a valid tracking sequence after multiple retries.")


def prepare_output_dirs(output_root: Path) -> tuple[Path, Path, Path]:
    sequences_root = output_root / "sequences"
    previews_root = output_root / "debug_preview"
    output_root.mkdir(parents=True, exist_ok=True)
    sequences_root.mkdir(parents=True, exist_ok=True)
    previews_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        split_dir = sequences_root / split
        if split_dir.exists():
            for item in split_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
        else:
            split_dir.mkdir(parents=True, exist_ok=True)

    for preview in previews_root.glob("*.png"):
        preview.unlink()

    return sequences_root, previews_root, output_root / "keyframes_manifest.csv"


def generate_tracking_dataset(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    output_root = args.output.resolve()
    sequences_root, previews_root, manifest_path = prepare_output_dirs(output_root)
    train_count = int(round(args.sequence_count * args.train_ratio))
    train_count = max(0, min(args.sequence_count, train_count))
    preview_indices = set(range(min(args.sequence_count, args.debug_preview_count)))

    manifest_rows: list[dict[str, object]] = []
    for seq_idx in range(args.sequence_count):
        split = "train" if seq_idx < train_count else "val"
        seq_name = f"seq_{seq_idx:04d}"
        seq_dir = sequences_root / split / seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)

        frames, meta, keyframes = build_tracking_sequence(
            image_size=args.image_size,
            frame_count=args.frames_per_sequence,
            rng=rng,
        )

        for frame_info, image in zip(keyframes, frames):
            frame_name = f"frame_{int(frame_info['frame_idx']):04d}.png"
            image.save(seq_dir / frame_name)
            manifest_rows.append(
                {
                    "sequence_id": seq_name,
                    "split": split,
                    "frame_idx": frame_info["frame_idx"],
                    "image": f"sequences/{split}/{seq_name}/{frame_name}",
                    "image_width": meta["image_width"],
                    "image_height": meta["image_height"],
                    "center_x": frame_info["center_x"],
                    "center_y": frame_info["center_y"],
                    "rotation_deg": frame_info["rotation_deg"],
                    "car_width": frame_info["car_width"],
                    "car_height": frame_info["car_height"],
                    "background_style": meta["background_style"],
                    "route_id": meta["route_id"],
                    "trajectory_type": meta["trajectory_type"],
                    "car_color": meta["car_color"],
                    "scale": meta["scale"],
                }
            )

        keyframe_payload = {
            "sequence_id": seq_name,
            "split": split,
            "image_width": meta["image_width"],
            "image_height": meta["image_height"],
            "background_style": meta["background_style"],
            "route_id": meta["route_id"],
            "trajectory_type": meta["trajectory_type"],
            "car_color": meta["car_color"],
            "scale": meta["scale"],
            "keyframes": keyframes,
        }
        with open(seq_dir / "keyframes.json", "w", encoding="utf-8") as f:
            json.dump(keyframe_payload, f, ensure_ascii=True, indent=2)

        if seq_idx in preview_indices and frames:
            preview = add_preview_marker(frames[0], float(keyframes[0]["center_x"]), float(keyframes[0]["center_y"]))
            preview.save(previews_root / f"{seq_name}.png")

    fieldnames = [
        "sequence_id",
        "split",
        "frame_idx",
        "image",
        "image_width",
        "image_height",
        "center_x",
        "center_y",
        "rotation_deg",
        "car_width",
        "car_height",
        "background_style",
        "route_id",
        "trajectory_type",
        "car_color",
        "scale",
    ]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)


def main() -> None:
    args = parse_args()
    if args.sequence_count <= 0:
        raise ValueError("--sequence-count must be a positive integer.")
    if args.frames_per_sequence <= 1:
        raise ValueError("--frames-per-sequence must be greater than 1.")
    if args.image_size < 128:
        raise ValueError("--image-size must be at least 128.")
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1.")
    generate_tracking_dataset(args)


if __name__ == "__main__":
    main()
