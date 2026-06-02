import argparse
import csv
import json
import math
import random
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from generate_dataset import (
    add_preview_marker,
    clear_pngs,
    make_vehicle_sprite,
    rotate_vehicle,
    validate_annotations,
)


BACKGROUND_STYLE = "dark_intersection_side_lines_only"
DEFAULT_RANDOM_RATIO = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a blacked-out fixed intersection dataset with side lane lines only. "
            "The output contains random-position still images and straight constant-speed sequences."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/dark_intersection_100k.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--total-count", type=int)
    parser.add_argument("--random-count", type=int)
    parser.add_argument("--sequence-count", type=int)
    parser.add_argument("--frames-per-sequence", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--train-ratio", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--safety-margin", type=int)
    parser.add_argument("--debug-preview-count", type=int)
    parser.add_argument("--car-color", type=str)
    parser.add_argument("--car-scale", type=float)
    parser.add_argument("--split-min-train-distance-px", type=float)
    parser.add_argument("--target-output-size", type=int)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merged_config(args: argparse.Namespace) -> dict[str, Any]:
    config = {
        "output": "dataset_dark_intersection",
        "total_count": 100000,
        "random_ratio": DEFAULT_RANDOM_RATIO,
        "frames_per_sequence": 20,
        "image_size": 640,
        "train_ratio": 0.8,
        "seed": 42,
        "safety_margin": 28,
        "debug_preview_count": 8,
        "car_color": "blue",
        "car_scale": 0.85,
        "sampling_mode": "without_replacement",
        "unique_coordinates": False,
        "road_color_mode": "normal",
        "lane_line_color_mode": "normal",
        "road_layout": "default",
        "random_region_mode": "lanes",
        "allow_partial_vehicle": False,
        "split_min_train_distance_px": 0.0,
        "background_style": BACKGROUND_STYLE,
        "target_types": ["box_mask"],
        "target_output_size": 128,
    }
    config.update(load_config(args.config))

    cli_overrides = {
        "output": args.output,
        "total_count": args.total_count,
        "random_count": args.random_count,
        "sequence_count": args.sequence_count,
        "frames_per_sequence": args.frames_per_sequence,
        "image_size": args.image_size,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "safety_margin": args.safety_margin,
        "debug_preview_count": args.debug_preview_count,
        "car_color": args.car_color,
        "car_scale": args.car_scale,
        "split_min_train_distance_px": args.split_min_train_distance_px,
        "target_output_size": args.target_output_size,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value

    frames_per_sequence = int(config["frames_per_sequence"])
    if "random_count" not in config:
        config["random_count"] = int(round(int(config["total_count"]) * float(config["random_ratio"])))
    if "sequence_count" not in config:
        sequence_frame_count = int(config["total_count"]) - int(config["random_count"])
        config["sequence_count"] = math.ceil(sequence_frame_count / frames_per_sequence)

    config["sequence_frame_count"] = int(config["sequence_count"]) * frames_per_sequence
    config["effective_total_count"] = int(config["random_count"]) + int(config["sequence_frame_count"])
    config["background_style"] = BACKGROUND_STYLE
    return config


def validate_config(config: dict[str, Any]) -> None:
    if int(config["random_count"]) <= 0:
        raise ValueError("random_count must be positive.")
    if int(config["sequence_count"]) < 0:
        raise ValueError("sequence_count must be non-negative.")
    if int(config["sequence_count"]) > 0 and int(config["frames_per_sequence"]) <= 1:
        raise ValueError("frames_per_sequence must be greater than 1.")
    if int(config["image_size"]) < 128:
        raise ValueError("image_size must be at least 128.")
    if not 0 < float(config["train_ratio"]) < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if int(config["safety_margin"]) < 0:
        raise ValueError("safety_margin must be non-negative.")
    if float(config["car_scale"]) <= 0:
        raise ValueError("car_scale must be positive.")
    if float(config.get("split_min_train_distance_px", 0.0)) < 0:
        raise ValueError("split_min_train_distance_px must be non-negative.")
    if str(config.get("sampling_mode", "without_replacement")) != "without_replacement":
        raise ValueError("This generator currently supports sampling_mode=without_replacement only.")
    if str(config.get("road_color_mode", "normal")) not in {"normal", "background"}:
        raise ValueError("road_color_mode must be one of: normal, background.")
    if str(config.get("lane_line_color_mode", "normal")) not in {"normal", "background"}:
        raise ValueError("lane_line_color_mode must be one of: normal, background.")
    if str(config.get("road_layout", "default")) not in {"default", "four_lane_bidirectional"}:
        raise ValueError("road_layout must be one of: default, four_lane_bidirectional.")
    if str(config.get("random_region_mode", "lanes")) not in {"lanes", "road_surface"}:
        raise ValueError("random_region_mode must be one of: lanes, road_surface.")
    if int(config.get("target_output_size", 128)) <= 0:
        raise ValueError("target_output_size must be positive.")


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def heading_from_vector(dx: float, dy: float) -> int:
    return int(round((math.degrees(math.atan2(dy, dx)) + 90.0) % 360.0))


def lerp_point(p0: tuple[float, float], p1: tuple[float, float], t: float) -> tuple[float, float]:
    return (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)


def image_key(rotation_deg: int, center_x: float, center_y: float) -> tuple[int, int, int]:
    return (int(rotation_deg) % 360, int(round(center_x)), int(round(center_y)))


def coordinate_key(key: tuple[int, int, int]) -> tuple[int, int]:
    return (int(key[1]), int(key[2]))


def use_unique_coordinates(config: dict[str, Any]) -> bool:
    return bool(config.get("unique_coordinates", False))


def add_train_key(
    train_points_by_rotation: dict[int, list[tuple[int, int]]],
    key: tuple[int, int, int],
) -> None:
    rotation_deg, center_x, center_y = key
    train_points_by_rotation.setdefault(int(rotation_deg) % 360, []).append((int(center_x), int(center_y)))


def add_train_lookup_key(
    train_lookup_by_rotation: dict[int, set[tuple[int, int]]],
    key: tuple[int, int, int],
) -> None:
    rotation_deg, center_x, center_y = key
    train_lookup_by_rotation.setdefault(int(rotation_deg) % 360, set()).add((int(center_x), int(center_y)))


def far_from_train_points(
    key: tuple[int, int, int],
    train_points_by_rotation: dict[int, list[tuple[int, int]]],
    min_distance_px: float,
) -> bool:
    if min_distance_px <= 0:
        return True
    rotation_deg, center_x, center_y = key
    points = train_points_by_rotation.get(int(rotation_deg) % 360, [])
    if not points:
        return True
    min_distance_sq = min_distance_px * min_distance_px
    return all((center_x - tx) * (center_x - tx) + (center_y - ty) * (center_y - ty) >= min_distance_sq for tx, ty in points)


def all_far_from_train_points(
    keys: list[tuple[int, int, int]],
    train_points_by_rotation: dict[int, list[tuple[int, int]]],
    min_distance_px: float,
) -> bool:
    return all(far_from_train_points(key, train_points_by_rotation, min_distance_px) for key in keys)


def far_from_train_lookup(
    key: tuple[int, int, int],
    train_lookup_by_rotation: dict[int, set[tuple[int, int]]],
    min_distance_px: float,
) -> bool:
    if min_distance_px <= 0:
        return True
    rotation_deg, center_x, center_y = key
    points = train_lookup_by_rotation.get(int(rotation_deg) % 360, set())
    if not points:
        return True
    radius = int(math.ceil(min_distance_px))
    min_distance_sq = min_distance_px * min_distance_px
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            tx = int(center_x) + dx
            ty = int(center_y) + dy
            if (tx, ty) in points and dx * dx + dy * dy < min_distance_sq:
                return False
    return True


def all_far_from_train_lookup(
    keys: list[tuple[int, int, int]],
    train_lookup_by_rotation: dict[int, set[tuple[int, int]]],
    min_distance_px: float,
) -> bool:
    return all(far_from_train_lookup(key, train_lookup_by_rotation, min_distance_px) for key in keys)


def road_geometry(image_size: int, config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or {}
    road_layout = str(config.get("road_layout", "default"))
    edge_margin = int(config.get("road_edge_margin_px", round(image_size * 0.055)))
    cx = image_size / 2.0
    cy = image_size / 2.0

    if road_layout == "default":
        road_w = int(round(image_size * 0.44))
        x0 = int(round(cx - road_w / 2.0))
        x1 = int(round(cx + road_w / 2.0))
        y0 = int(round(cy - road_w / 2.0))
        y1 = int(round(cy + road_w / 2.0))
        lane_offset = int(round(road_w * 0.22))
        lane_half_width = int(round(road_w * 0.11))
        lane_regions = [
            {
                "bbox": (
                    int(cx + lane_offset - lane_half_width),
                    edge_margin,
                    int(cx + lane_offset + lane_half_width),
                    image_size - edge_margin,
                ),
                "headings": (0, 180),
                "route_id": "north_to_south",
                "travel_axis": "y",
                "lateral_range": (int(cx + lane_offset - lane_half_width), int(cx + lane_offset + lane_half_width)),
                "travel_range": (edge_margin, image_size - edge_margin),
            },
            {
                "bbox": (
                    int(cx - lane_offset - lane_half_width),
                    edge_margin,
                    int(cx - lane_offset + lane_half_width),
                    image_size - edge_margin,
                ),
                "headings": (0, 180),
                "route_id": "south_to_north",
                "travel_axis": "y",
                "lateral_range": (int(cx - lane_offset - lane_half_width), int(cx - lane_offset + lane_half_width)),
                "travel_range": (edge_margin, image_size - edge_margin),
            },
            {
                "bbox": (
                    edge_margin,
                    int(cy - lane_offset - lane_half_width),
                    image_size - edge_margin,
                    int(cy - lane_offset + lane_half_width),
                ),
                "headings": (90, 270),
                "route_id": "west_to_east",
                "travel_axis": "x",
                "lateral_range": (int(cy - lane_offset - lane_half_width), int(cy - lane_offset + lane_half_width)),
                "travel_range": (edge_margin, image_size - edge_margin),
            },
            {
                "bbox": (
                    edge_margin,
                    int(cy + lane_offset - lane_half_width),
                    image_size - edge_margin,
                    int(cy + lane_offset + lane_half_width),
                ),
                "headings": (90, 270),
                "route_id": "east_to_west",
                "travel_axis": "x",
                "lateral_range": (int(cy + lane_offset - lane_half_width), int(cy + lane_offset + lane_half_width)),
                "travel_range": (edge_margin, image_size - edge_margin),
            },
        ]
        line_positions = {
            "vertical": [x0 + int(round(image_size * 0.035)), x1 - int(round(image_size * 0.035))],
            "horizontal": [y0 + int(round(image_size * 0.035)), y1 - int(round(image_size * 0.035))],
        }
        return {
            "road_layout": road_layout,
            "cx": cx,
            "cy": cy,
            "x0": x0,
            "x1": x1,
            "y0": y0,
            "y1": y1,
            "edge_margin": edge_margin,
            "lane_regions": lane_regions,
            "line_positions": line_positions,
        }

    road_w = int(round(image_size * float(config.get("road_width_ratio_four_lane", 0.56))))
    x0 = int(round(cx - road_w / 2.0))
    x1 = int(round(cx + road_w / 2.0))
    y0 = int(round(cy - road_w / 2.0))
    y1 = int(round(cy + road_w / 2.0))
    lane_width = road_w / 4.0
    lane_half_width = int(round(lane_width * 0.34))
    vertical_centers = [cx - 1.5 * lane_width, cx - 0.5 * lane_width, cx + 0.5 * lane_width, cx + 1.5 * lane_width]
    horizontal_centers = [cy - 1.5 * lane_width, cy - 0.5 * lane_width, cy + 0.5 * lane_width, cy + 1.5 * lane_width]

    lane_regions: list[dict[str, object]] = []
    vertical_specs = [
        ("south_to_north_outer", vertical_centers[0], (0, 180)),
        ("south_to_north_inner", vertical_centers[1], (0, 180)),
        ("north_to_south_inner", vertical_centers[2], (0, 180)),
        ("north_to_south_outer", vertical_centers[3], (0, 180)),
    ]
    for route_id, center_x, headings in vertical_specs:
        lane_regions.append(
            {
                "bbox": (
                    int(round(center_x - lane_half_width)),
                    edge_margin,
                    int(round(center_x + lane_half_width)),
                    image_size - edge_margin,
                ),
                "headings": headings,
                "route_id": route_id,
                "travel_axis": "y",
                "lateral_range": (int(round(center_x - lane_half_width)), int(round(center_x + lane_half_width))),
                "travel_range": (edge_margin, image_size - edge_margin),
            }
        )
    horizontal_specs = [
        ("west_to_east_outer", horizontal_centers[0], (90, 270)),
        ("west_to_east_inner", horizontal_centers[1], (90, 270)),
        ("east_to_west_inner", horizontal_centers[2], (90, 270)),
        ("east_to_west_outer", horizontal_centers[3], (90, 270)),
    ]
    for route_id, center_y, headings in horizontal_specs:
        lane_regions.append(
            {
                "bbox": (
                    edge_margin,
                    int(round(center_y - lane_half_width)),
                    image_size - edge_margin,
                    int(round(center_y + lane_half_width)),
                ),
                "headings": headings,
                "route_id": route_id,
                "travel_axis": "x",
                "lateral_range": (int(round(center_y - lane_half_width)), int(round(center_y + lane_half_width))),
                "travel_range": (edge_margin, image_size - edge_margin),
            }
        )
    line_positions = {
        "vertical": [int(round(x0 + lane_width * idx)) for idx in range(1, 4)],
        "horizontal": [int(round(y0 + lane_width * idx)) for idx in range(1, 4)],
    }
    return {
        "road_layout": road_layout,
        "cx": cx,
        "cy": cy,
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
        "edge_margin": edge_margin,
        "lane_regions": lane_regions,
        "line_positions": line_positions,
    }


def make_dark_intersection_background(
    image_size: int,
    config: dict[str, Any] | None = None,
) -> tuple[Image.Image, list[dict[str, object]], list[dict[str, object]]]:
    config = config or {}
    bg_color = (2, 3, 5)
    road_color = bg_color if str(config.get("road_color_mode", "normal")) == "background" else (25, 26, 29)
    line_color = bg_color if str(config.get("lane_line_color_mode", "normal")) == "background" else (104, 108, 112)

    image = Image.new("RGB", (image_size, image_size), bg_color)
    draw = ImageDraw.Draw(image)
    geometry = road_geometry(image_size, config)
    x0 = int(geometry["x0"])
    x1 = int(geometry["x1"])
    y0 = int(geometry["y0"])
    y1 = int(geometry["y1"])

    draw.rectangle((x0, 0, x1, image_size), fill=road_color)
    draw.rectangle((0, y0, image_size, y1), fill=road_color)

    side_line_w = max(3, int(round(image_size * 0.006)))
    for x in [int(pos) - side_line_w // 2 for pos in geometry["line_positions"]["vertical"]]:
        draw.rectangle((x, 0, x + side_line_w, y0 - 1), fill=line_color)
        draw.rectangle((x, y1 + 1, x + side_line_w, image_size), fill=line_color)
    for y in [int(pos) - side_line_w // 2 for pos in geometry["line_positions"]["horizontal"]]:
        draw.rectangle((0, y, x0 - 1, y + side_line_w), fill=line_color)
        draw.rectangle((x1 + 1, y, image_size, y + side_line_w), fill=line_color)

    regions = []
    routes = []
    for lane in geometry["lane_regions"]:
        regions.append({"bbox": lane["bbox"], "headings": lane["headings"], "jitter_deg": 0})
        lateral_min, lateral_max = tuple(lane["lateral_range"])
        travel_min, travel_max = tuple(lane["travel_range"])
        center_lateral = (float(lateral_min) + float(lateral_max)) / 2.0
        if str(lane["travel_axis"]) == "y":
            starts_at_top = "north_to_south" in str(lane["route_id"])
            start = (center_lateral, float(travel_min if starts_at_top else travel_max))
            end = (center_lateral, float(travel_max if starts_at_top else travel_min))
        else:
            starts_at_left = "west_to_east" in str(lane["route_id"])
            start = (float(travel_min if starts_at_left else travel_max), center_lateral)
            end = (float(travel_max if starts_at_left else travel_min), center_lateral)
        routes.append(
            {
                "route_id": lane["route_id"],
                "start": start,
                "end": end,
                "travel_axis": lane["travel_axis"],
                "lateral_range": lane["lateral_range"],
                "travel_range": lane["travel_range"],
            }
        )
    return image, regions, routes


def clamp_center(center: float, radius: float, low: float, high: float) -> float:
    return max(low + radius, min(high - radius, center))


@lru_cache(maxsize=32)
def vehicle_dimensions(car_scale: float, car_color: str, rotation_deg: int) -> tuple[int, int]:
    sprite, vehicle_mask = make_vehicle_sprite(scale=car_scale, color_name=car_color)
    _, vehicle_bbox = rotate_vehicle(sprite, vehicle_mask, int(rotation_deg))
    return vehicle_bbox[2] - vehicle_bbox[0], vehicle_bbox[3] - vehicle_bbox[1]


def draw_vehicle(
    background: Image.Image,
    sprite: Image.Image,
    vehicle_mask: Image.Image,
    center: tuple[float, float],
    rotation_deg: int,
    image_size: int,
    safety_margin: int,
    allow_partial_vehicle: bool = False,
) -> tuple[Image.Image, dict[str, object]]:
    rotated_sprite, vehicle_bbox = rotate_vehicle(sprite, vehicle_mask, rotation_deg)
    vehicle_width = vehicle_bbox[2] - vehicle_bbox[0]
    vehicle_height = vehicle_bbox[3] - vehicle_bbox[1]
    if allow_partial_vehicle:
        cx = float(center[0])
        cy = float(center[1])
    else:
        cx = clamp_center(center[0], vehicle_width / 2.0, safety_margin, image_size - safety_margin)
        cy = clamp_center(center[1], vehicle_height / 2.0, safety_margin, image_size - safety_margin)

    bbox_x = cx - vehicle_width / 2.0
    bbox_y = cy - vehicle_height / 2.0
    paste_x = int(round(bbox_x - vehicle_bbox[0]))
    paste_y = int(round(bbox_y - vehicle_bbox[1]))
    canvas = background.convert("RGBA")
    if allow_partial_vehicle:
        crop_x0 = max(0, paste_x)
        crop_y0 = max(0, paste_y)
        crop_x1 = min(image_size, paste_x + rotated_sprite.size[0])
        crop_y1 = min(image_size, paste_y + rotated_sprite.size[1])
        if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
            raise RuntimeError("Vehicle placement has no visible area inside the image bounds.")
        sprite_crop = rotated_sprite.crop((crop_x0 - paste_x, crop_y0 - paste_y, crop_x1 - paste_x, crop_y1 - paste_y))
        canvas.alpha_composite(sprite_crop, dest=(crop_x0, crop_y0))
    else:
        if (
            paste_x < 0
            or paste_y < 0
            or paste_x + rotated_sprite.size[0] > image_size
            or paste_y + rotated_sprite.size[1] > image_size
        ):
            raise RuntimeError("Vehicle placement falls outside the image bounds.")
        canvas.alpha_composite(rotated_sprite, dest=(paste_x, paste_y))
    metadata = {
        "center_x": round(cx, 2),
        "center_y": round(cy, 2),
        "car_width": vehicle_width,
        "car_height": vehicle_height,
        "rotation_deg": rotation_deg,
    }
    return canvas.convert("RGB"), metadata


def target_types_enabled(config: dict[str, Any]) -> set[str]:
    target_types = config.get("target_types", [])
    if isinstance(target_types, str):
        target_types = [target_types]
    return {str(t).strip().lower() for t in target_types if str(t).strip()}


def make_box_mask_target(
    metadata: dict[str, object],
    image_size: int,
    target_output_size: int,
) -> Image.Image:
    """Create a 128x128-style target where the vehicle exterior box is bright."""
    scale = float(target_output_size) / float(image_size)
    cx = float(metadata["center_x"]) * scale
    cy = float(metadata["center_y"]) * scale
    w = float(metadata["car_width"]) * scale
    h = float(metadata["car_height"]) * scale

    x0 = int(round(cx - w / 2.0))
    y0 = int(round(cy - h / 2.0))
    x1 = int(round(cx + w / 2.0))
    y1 = int(round(cy + h / 2.0))
    x0 = max(0, min(target_output_size - 1, x0))
    y0 = max(0, min(target_output_size - 1, y0))
    x1 = max(0, min(target_output_size - 1, x1))
    y1 = max(0, min(target_output_size - 1, y1))

    target = Image.new("L", (target_output_size, target_output_size), 0)
    draw = ImageDraw.Draw(target)
    draw.rectangle((x0, y0, x1, y1), fill=255)
    return target


def save_box_mask_target(
    *,
    dataset_root: Path,
    relative_image_path: Path,
    metadata: dict[str, object],
    config: dict[str, Any],
) -> str:
    target_output_size = int(config.get("target_output_size", 128))
    target_relative_path = Path("targets") / "box_mask" / relative_image_path
    target_path = dataset_root / target_relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target = make_box_mask_target(
        metadata=metadata,
        image_size=int(config["image_size"]),
        target_output_size=target_output_size,
    )
    target.save(target_path)
    return target_relative_path.as_posix()


def add_box_mask_preview(
    image: Image.Image,
    metadata: dict[str, object],
    config: dict[str, Any],
) -> Image.Image:
    target_output_size = int(config.get("target_output_size", 128))
    small = image.resize((target_output_size, target_output_size), Image.Resampling.BICUBIC).convert("RGB")
    mask = make_box_mask_target(metadata, int(config["image_size"]), target_output_size)
    overlay = Image.new("RGB", small.size, (30, 210, 120))
    alpha = mask.point(lambda v: int(v * 0.45))
    return Image.composite(overlay, small, alpha)


def build_random_candidates(
    image_size: int,
    safety_margin: int,
    car_color: str,
    car_scale: float,
    used_keys: set[tuple[int, int, int]],
    used_coords: set[tuple[int, int]] | None = None,
    random_region_mode: str = "lanes",
    allow_partial_vehicle: bool = False,
    config: dict[str, Any] | None = None,
) -> list[tuple[int, int, int]]:
    config = config or {}
    _, regions, _ = make_dark_intersection_background(image_size, config)
    if random_region_mode == "road_surface":
        geometry = road_geometry(image_size, config)
        x0 = int(geometry["x0"])
        x1 = int(geometry["x1"])
        y0 = int(geometry["y0"])
        y1 = int(geometry["y1"])
        edge_margin = int(geometry["edge_margin"])
        regions = [
            {
                "bbox": (x0, edge_margin, x1, image_size - edge_margin),
                "headings": (0, 180),
            },
            {
                "bbox": (edge_margin, y0, image_size - edge_margin, y1),
                "headings": (90, 270),
            },
        ]
    elif random_region_mode != "lanes":
        raise ValueError(f"Unsupported random_region_mode: {random_region_mode}")
    sprite, vehicle_mask = make_vehicle_sprite(scale=car_scale, color_name=car_color)
    candidates: list[tuple[int, int, int]] = []
    used_coords = used_coords or set()
    for region in regions:
        x0, y0, x1, y1 = tuple(region["bbox"])
        for rotation_deg in tuple(region["headings"]):
            _, vehicle_bbox = rotate_vehicle(sprite, vehicle_mask, int(rotation_deg))
            vehicle_width = vehicle_bbox[2] - vehicle_bbox[0]
            vehicle_height = vehicle_bbox[3] - vehicle_bbox[1]
            if allow_partial_vehicle:
                min_x = math.ceil(x0)
                max_x = math.floor(x1)
                min_y = math.ceil(y0)
                max_y = math.floor(y1)
            else:
                min_x = math.ceil(max(x0 + vehicle_width / 2.0, safety_margin + vehicle_width / 2.0))
                max_x = math.floor(min(x1 - vehicle_width / 2.0, image_size - safety_margin - vehicle_width / 2.0))
                min_y = math.ceil(max(y0 + vehicle_height / 2.0, safety_margin + vehicle_height / 2.0))
                max_y = math.floor(min(y1 - vehicle_height / 2.0, image_size - safety_margin - vehicle_height / 2.0))
            if max_x < min_x or max_y < min_y:
                continue
            for center_x in range(min_x, max_x + 1):
                for center_y in range(min_y, max_y + 1):
                    key = image_key(int(rotation_deg), center_x, center_y)
                    if key not in used_keys and coordinate_key(key) not in used_coords:
                        candidates.append(key)
    return candidates


def keep_one_heading_per_coordinate(
    candidates: list[tuple[int, int, int]],
    rng: random.Random,
) -> list[tuple[int, int, int]]:
    shuffled = candidates[:]
    rng.shuffle(shuffled)
    seen_coords: set[tuple[int, int]] = set()
    unique: list[tuple[int, int, int]] = []
    for key in shuffled:
        coord = coordinate_key(key)
        if coord in seen_coords:
            continue
        seen_coords.add(coord)
        unique.append(key)
    return unique


def build_random_sample(
    config: dict[str, Any],
    image_size: int,
    safety_margin: int,
    car_color: str,
    car_scale: float,
    sample_key: tuple[int, int, int],
) -> tuple[Image.Image, dict[str, object]]:
    background, _, _ = make_dark_intersection_background(image_size, config)
    sprite, vehicle_mask = make_vehicle_sprite(scale=car_scale, color_name=car_color)
    rotation_deg, center_x, center_y = sample_key
    image, placement = draw_vehicle(
        background=background,
        sprite=sprite,
        vehicle_mask=vehicle_mask,
        center=(float(center_x), float(center_y)),
        rotation_deg=rotation_deg,
        image_size=image_size,
        safety_margin=safety_margin,
        allow_partial_vehicle=bool(config.get("allow_partial_vehicle", False)),
    )
    canvas = background.convert("RGBA")
    metadata = {
        "center_x": placement["center_x"],
        "center_y": placement["center_y"],
        "car_width": placement["car_width"],
        "car_height": placement["car_height"],
        "rotation_deg": rotation_deg,
        "scale": car_scale,
        "background_style": BACKGROUND_STYLE,
        "car_color": car_color,
        "sample_type": "random_lane_position",
    }
    return image, metadata


def generate_random_dataset(
    config: dict[str, Any],
    output_root: Path,
    rng: random.Random,
    used_keys: set[tuple[int, int, int]],
    used_coords: set[tuple[int, int]],
) -> set[tuple[int, int, int]]:
    random_root = output_root / "random"
    train_dir = random_root / "images" / "train"
    val_dir = random_root / "images" / "val"
    preview_dir = random_root / "debug_preview"
    target_types = target_types_enabled(config)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    clear_pngs(train_dir)
    clear_pngs(val_dir)
    clear_pngs(preview_dir)

    count = int(config["random_count"])
    train_count = int(round(count * float(config["train_ratio"])))
    debug_count = max(0, min(count, int(config["debug_preview_count"])))
    debug_indices = set(random.Random(int(config["seed"]) + 999).sample(range(count), k=debug_count)) if debug_count else set()
    candidates = build_random_candidates(
        image_size=int(config["image_size"]),
        safety_margin=int(config["safety_margin"]),
        car_color=str(config["car_color"]),
        car_scale=float(config["car_scale"]),
        used_keys=used_keys,
        used_coords=used_coords if use_unique_coordinates(config) else None,
        random_region_mode=str(config.get("random_region_mode", "lanes")),
        allow_partial_vehicle=bool(config.get("allow_partial_vehicle", False)),
        config=config,
    )
    if use_unique_coordinates(config):
        candidates = keep_one_heading_per_coordinate(candidates, rng)
    if count > len(candidates):
        raise ValueError(f"Requested {count} random samples, but only {len(candidates)} unique random placements are available.")
    selected_keys = rng.sample(candidates, k=count)

    records: list[dict[str, object]] = []
    for index in range(count):
        split = "train" if index < train_count else "val"
        image, metadata = build_random_sample(
            config=config,
            image_size=int(config["image_size"]),
            safety_margin=int(config["safety_margin"]),
            car_color=str(config["car_color"]),
            car_scale=float(config["car_scale"]),
            sample_key=selected_keys[index],
        )
        used_keys.add(selected_keys[index])
        used_coords.add(coordinate_key(selected_keys[index]))
        filename = f"sample_{index:06d}.png"
        relative_path = Path("images") / split / filename
        target_dir = train_dir if split == "train" else val_dir
        image.save(target_dir / filename)
        record = {
            "image": relative_path.as_posix(),
            "split": split,
            "image_width": int(config["image_size"]),
            "image_height": int(config["image_size"]),
            **metadata,
        }
        if "box_mask" in target_types:
            record["box_mask"] = save_box_mask_target(
                dataset_root=random_root,
                relative_image_path=relative_path,
                metadata=metadata,
                config=config,
            )
        records.append(record)
        if index in debug_indices:
            add_preview_marker(image, float(metadata["center_x"]), float(metadata["center_y"])).save(preview_dir / filename)
            if "box_mask" in target_types:
                add_box_mask_preview(image, metadata, config).save(preview_dir / f"{Path(filename).stem}_box_mask_overlay.png")

    validate_annotations(
        records,
        int(config["image_size"]),
        train_count,
        count,
        allow_out_of_bounds_center=bool(config.get("allow_partial_vehicle", False)),
    )
    write_csv(random_root / "annotations.csv", records)
    return used_keys


def prepare_random_output(random_root: Path, target_types: set[str]) -> tuple[Path, Path, Path]:
    train_dir = random_root / "images" / "train"
    val_dir = random_root / "images" / "val"
    preview_dir = random_root / "debug_preview"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    clear_pngs(train_dir)
    clear_pngs(val_dir)
    clear_pngs(preview_dir)
    if "box_mask" in target_types:
        reset_dir(random_root / "targets" / "box_mask")
    return train_dir, val_dir, preview_dir


def append_random_record(
    config: dict[str, Any],
    random_root: Path,
    target_types: set[str],
    sample_key: tuple[int, int, int],
    split: str,
    index: int,
    debug_indices: set[int],
    preview_dir: Path,
    records: list[dict[str, object]],
) -> None:
    image, metadata = build_random_sample(
        config=config,
        image_size=int(config["image_size"]),
        safety_margin=int(config["safety_margin"]),
        car_color=str(config["car_color"]),
        car_scale=float(config["car_scale"]),
        sample_key=sample_key,
    )
    filename = f"sample_{index:06d}.png"
    relative_path = Path("images") / split / filename
    target_dir = random_root / "images" / split
    image.save(target_dir / filename)
    record = {
        "image": relative_path.as_posix(),
        "split": split,
        "image_width": int(config["image_size"]),
        "image_height": int(config["image_size"]),
        **metadata,
    }
    if "box_mask" in target_types:
        record["box_mask"] = save_box_mask_target(
            dataset_root=random_root,
            relative_image_path=relative_path,
            metadata=metadata,
            config=config,
        )
    records.append(record)
    if index in debug_indices:
        add_preview_marker(image, float(metadata["center_x"]), float(metadata["center_y"])).save(preview_dir / filename)
        if "box_mask" in target_types:
            add_box_mask_preview(image, metadata, config).save(preview_dir / f"{Path(filename).stem}_box_mask_overlay.png")


def generate_random_dataset_isolated(
    config: dict[str, Any],
    output_root: Path,
    rng: random.Random,
    used_keys: set[tuple[int, int, int]],
    used_coords: set[tuple[int, int]],
    train_points_by_rotation: dict[int, list[tuple[int, int]]],
) -> None:
    random_root = output_root / "random"
    target_types = target_types_enabled(config)
    _, _, preview_dir = prepare_random_output(random_root, target_types)

    count = int(config["random_count"])
    train_count = int(round(count * float(config["train_ratio"])))
    val_count = count - train_count
    min_distance_px = float(config.get("split_min_train_distance_px", 0.0))
    debug_count = max(0, min(count, int(config["debug_preview_count"])))
    debug_indices = set(random.Random(int(config["seed"]) + 999).sample(range(count), k=debug_count)) if debug_count else set()
    candidates = build_random_candidates(
        image_size=int(config["image_size"]),
        safety_margin=int(config["safety_margin"]),
        car_color=str(config["car_color"]),
        car_scale=float(config["car_scale"]),
        used_keys=used_keys,
        used_coords=used_coords if use_unique_coordinates(config) else None,
        random_region_mode=str(config.get("random_region_mode", "lanes")),
        allow_partial_vehicle=bool(config.get("allow_partial_vehicle", False)),
        config=config,
    )
    if use_unique_coordinates(config):
        candidates = keep_one_heading_per_coordinate(candidates, rng)
    if train_count > len(candidates):
        raise ValueError(f"Requested {train_count} random train samples, but only {len(candidates)} unique random placements are available.")

    shuffled = rng.sample(candidates, k=len(candidates))
    if use_unique_coordinates(config) and min_distance_px > 0:
        val_keys = shuffled[:val_count]
        val_points_by_rotation: dict[int, list[tuple[int, int]]] = {}
        val_key_set = set(val_keys)
        val_coord_set = {coordinate_key(key) for key in val_keys}
        for key in val_keys:
            add_train_key(val_points_by_rotation, key)
        train_pool = [
            key
            for key in shuffled[val_count:]
            if key not in val_key_set
            and coordinate_key(key) not in val_coord_set
            and far_from_train_points(key, val_points_by_rotation, min_distance_px)
        ]
        if train_count > len(train_pool):
            raise ValueError(
                f"Requested {train_count} random train samples with split_min_train_distance_px={min_distance_px}, "
                f"but only {len(train_pool)} placements remain after reserving isolated validation samples."
            )
        train_keys = train_pool[:train_count]
    else:
        train_keys = shuffled[:train_count]
        candidate_train_points_by_rotation = {rotation: points.copy() for rotation, points in train_points_by_rotation.items()}
        for key in train_keys:
            add_train_key(candidate_train_points_by_rotation, key)
        val_pool = [
            key
            for key in shuffled[train_count:]
            if key not in used_keys
            and (not use_unique_coordinates(config) or coordinate_key(key) not in used_coords)
            and far_from_train_points(key, candidate_train_points_by_rotation, min_distance_px)
        ]
        if val_count > len(val_pool):
            raise ValueError(
                f"Requested {val_count} random val samples with split_min_train_distance_px={min_distance_px}, "
                f"but only {len(val_pool)} isolated placements are available."
            )
        val_keys = val_pool[:val_count]

    for key in train_keys:
        used_keys.add(key)
        used_coords.add(coordinate_key(key))
        add_train_key(train_points_by_rotation, key)
    for key in val_keys:
        used_keys.add(key)
        used_coords.add(coordinate_key(key))

    records: list[dict[str, object]] = []
    for index, key in enumerate(train_keys + val_keys):
        split = "train" if index < train_count else "val"
        append_random_record(config, random_root, target_types, key, split, index, debug_indices, preview_dir, records)

    validate_annotations(
        records,
        int(config["image_size"]),
        train_count,
        count,
        allow_out_of_bounds_center=bool(config.get("allow_partial_vehicle", False)),
    )
    write_csv(random_root / "annotations.csv", records)


def build_straight_sequence(
    config: dict[str, Any],
    rng: random.Random,
    used_keys: set[tuple[int, int, int]],
    used_coords: set[tuple[int, int]],
    train_points_by_rotation: dict[int, list[tuple[int, int]]] | None = None,
    min_train_distance_px: float = 0.0,
) -> tuple[list[Image.Image], dict[str, object], list[dict[str, object]]]:
    image_size = int(config["image_size"])
    background, _, routes = make_dark_intersection_background(image_size, config)
    sprite, vehicle_mask = make_vehicle_sprite(scale=float(config["car_scale"]), color_name=str(config["car_color"]))
    frame_count = int(config["frames_per_sequence"])
    max_trim = max(0, int(round(image_size * 0.08)))

    for _ in range(10_000):
        route = rng.choice(routes)
        axis = str(route["travel_axis"])
        lateral_min, lateral_max = tuple(route["lateral_range"])
        travel_min, travel_max = tuple(route["travel_range"])
        lateral = rng.randint(int(lateral_min), int(lateral_max))
        trim0 = rng.randint(0, max_trim)
        trim1 = rng.randint(0, max_trim)

        if axis == "y":
            if tuple(route["start"])[1] < tuple(route["end"])[1]:
                start = (float(lateral), float(travel_min + trim0))
                end = (float(lateral), float(travel_max - trim1))
            else:
                start = (float(lateral), float(travel_max - trim0))
                end = (float(lateral), float(travel_min + trim1))
        else:
            if tuple(route["start"])[0] < tuple(route["end"])[0]:
                start = (float(travel_min + trim0), float(lateral))
                end = (float(travel_max - trim1), float(lateral))
            else:
                start = (float(travel_max - trim0), float(lateral))
                end = (float(travel_min + trim1), float(lateral))

        heading = heading_from_vector(end[0] - start[0], end[1] - start[1])
        points = [lerp_point(start, end, idx / max(1, frame_count - 1)) for idx in range(frame_count)]
        frames = []
        keyframes = []
        for frame_idx, point in enumerate(points):
            image, frame_meta = draw_vehicle(
                background=background,
                sprite=sprite,
                vehicle_mask=vehicle_mask,
                center=point,
                rotation_deg=heading,
                image_size=image_size,
                safety_margin=int(config["safety_margin"]),
                allow_partial_vehicle=bool(config.get("allow_partial_vehicle", False)),
            )
            frames.append(image)
            keyframes.append({"frame_idx": frame_idx, **frame_meta})

        frame_keys = [
            image_key(int(frame["rotation_deg"]), float(frame["center_x"]), float(frame["center_y"]))
            for frame in keyframes
        ]
        if len(set(frame_keys)) != frame_count or any(key in used_keys for key in frame_keys):
            continue
        frame_coords = [coordinate_key(key) for key in frame_keys]
        if use_unique_coordinates(config) and (len(set(frame_coords)) != frame_count or any(coord in used_coords for coord in frame_coords)):
            continue
        if train_points_by_rotation is not None and not all_far_from_train_points(frame_keys, train_points_by_rotation, min_train_distance_px):
            continue

        used_keys.update(frame_keys)
        used_coords.update(frame_coords)
        meta = {
            "background_style": BACKGROUND_STYLE,
            "route_id": route["route_id"],
            "trajectory_type": "straight_constant_speed_without_replacement",
            "car_color": str(config["car_color"]),
            "scale": float(config["car_scale"]),
            "image_width": image_size,
            "image_height": image_size,
            "start_x": round(start[0], 2),
            "start_y": round(start[1], 2),
            "end_x": round(end[0], 2),
            "end_y": round(end[1], 2),
        }
        return frames, meta, keyframes

    raise RuntimeError("Failed to build a unique straight sequence after 10000 retries.")


def build_straight_candidate_metadata(
    config: dict[str, Any],
    route: dict[str, object],
    lateral: int,
    trim0: int,
    trim1: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[tuple[int, int, int]]]:
    image_size = int(config["image_size"])
    frame_count = int(config["frames_per_sequence"])
    axis = str(route["travel_axis"])
    travel_min, travel_max = tuple(route["travel_range"])

    if axis == "y":
        if tuple(route["start"])[1] < tuple(route["end"])[1]:
            start = (float(lateral), float(travel_min + trim0))
            end = (float(lateral), float(travel_max - trim1))
        else:
            start = (float(lateral), float(travel_max - trim0))
            end = (float(lateral), float(travel_min + trim1))
    else:
        if tuple(route["start"])[0] < tuple(route["end"])[0]:
            start = (float(travel_min + trim0), float(lateral))
            end = (float(travel_max - trim1), float(lateral))
        else:
            start = (float(travel_max - trim0), float(lateral))
            end = (float(travel_min + trim1), float(lateral))

    heading = heading_from_vector(end[0] - start[0], end[1] - start[1])
    vehicle_width, vehicle_height = vehicle_dimensions(float(config["car_scale"]), str(config["car_color"]), heading)
    keyframes: list[dict[str, object]] = []
    frame_keys: list[tuple[int, int, int]] = []
    for frame_idx in range(frame_count):
        point = lerp_point(start, end, frame_idx / max(1, frame_count - 1))
        if bool(config.get("allow_partial_vehicle", False)):
            cx = point[0]
            cy = point[1]
        else:
            cx = clamp_center(point[0], vehicle_width / 2.0, int(config["safety_margin"]), image_size - int(config["safety_margin"]))
            cy = clamp_center(point[1], vehicle_height / 2.0, int(config["safety_margin"]), image_size - int(config["safety_margin"]))
        frame_meta = {
            "frame_idx": frame_idx,
            "center_x": round(cx, 2),
            "center_y": round(cy, 2),
            "car_width": vehicle_width,
            "car_height": vehicle_height,
            "rotation_deg": heading,
        }
        keyframes.append(frame_meta)
        frame_keys.append(image_key(heading, cx, cy))

    meta = {
        "background_style": BACKGROUND_STYLE,
        "route_id": route["route_id"],
        "trajectory_type": "straight_constant_speed_without_replacement",
        "car_color": str(config["car_color"]),
        "scale": float(config["car_scale"]),
        "image_width": image_size,
        "image_height": image_size,
        "start_x": round(start[0], 2),
        "start_y": round(start[1], 2),
        "end_x": round(end[0], 2),
        "end_y": round(end[1], 2),
    }
    return meta, keyframes, frame_keys


def render_straight_frames(
    config: dict[str, Any],
    keyframes: list[dict[str, object]],
) -> list[Image.Image]:
    image_size = int(config["image_size"])
    background, _, _ = make_dark_intersection_background(image_size, config)
    sprite, vehicle_mask = make_vehicle_sprite(scale=float(config["car_scale"]), color_name=str(config["car_color"]))
    frames: list[Image.Image] = []
    for frame in keyframes:
        image, _ = draw_vehicle(
            background=background,
            sprite=sprite,
            vehicle_mask=vehicle_mask,
            center=(float(frame["center_x"]), float(frame["center_y"])),
            rotation_deg=int(frame["rotation_deg"]),
            image_size=image_size,
            safety_margin=int(config["safety_margin"]),
            allow_partial_vehicle=bool(config.get("allow_partial_vehicle", False)),
        )
        frames.append(image)
    return frames


def build_shuffled_straight_params(
    config: dict[str, Any],
    rng: random.Random,
) -> list[tuple[dict[str, object], int, int, int]]:
    image_size = int(config["image_size"])
    _, _, routes = make_dark_intersection_background(image_size, config)
    max_trim = max(0, int(round(image_size * 0.08)))
    params: list[tuple[dict[str, object], int, int, int]] = []
    for route in routes:
        lateral_min, lateral_max = tuple(route["lateral_range"])
        for lateral in range(int(lateral_min), int(lateral_max) + 1):
            for trim0 in range(max_trim + 1):
                for trim1 in range(max_trim + 1):
                    params.append((route, lateral, trim0, trim1))
    rng.shuffle(params)
    return params


def generate_straight_dataset(
    config: dict[str, Any],
    output_root: Path,
    rng: random.Random,
    used_keys: set[tuple[int, int, int]],
    used_coords: set[tuple[int, int]],
) -> None:
    straight_root = output_root / "straight"
    sequences_root = straight_root / "sequences"
    preview_dir = straight_root / "debug_preview"
    target_types = target_types_enabled(config)
    reset_dir(sequences_root)
    reset_dir(preview_dir)

    sequence_count = int(config["sequence_count"])
    train_count = int(round(sequence_count * float(config["train_ratio"])))
    preview_indices = set(range(min(sequence_count, int(config["debug_preview_count"]))))

    manifest_rows: list[dict[str, object]] = []
    for seq_idx in range(sequence_count):
        split = "train" if seq_idx < train_count else "val"
        seq_name = f"seq_{seq_idx:05d}"
        seq_dir = sequences_root / split / seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)
        frames, meta, keyframes = build_straight_sequence(config, rng, used_keys, used_coords)

        for frame_info, image in zip(keyframes, frames):
            frame_name = f"frame_{int(frame_info['frame_idx']):04d}.png"
            relative_image_path = Path("sequences") / split / seq_name / frame_name
            image.save(seq_dir / frame_name)
            row = {
                "sequence_id": seq_name,
                "split": split,
                "frame_idx": frame_info["frame_idx"],
                "image": relative_image_path.as_posix(),
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
                "start_x": meta["start_x"],
                "start_y": meta["start_y"],
                "end_x": meta["end_x"],
                "end_y": meta["end_y"],
            }
            if "box_mask" in target_types:
                row["box_mask"] = save_box_mask_target(
                    dataset_root=straight_root,
                    relative_image_path=relative_image_path,
                    metadata=frame_info,
                    config=config,
                )
            manifest_rows.append(row)

        with open(seq_dir / "keyframes.json", "w", encoding="utf-8") as f:
            json.dump({**meta, "sequence_id": seq_name, "split": split, "keyframes": keyframes}, f, ensure_ascii=True, indent=2)
        if seq_idx in preview_indices and frames:
            first = keyframes[0]
            add_preview_marker(frames[0], float(first["center_x"]), float(first["center_y"])).save(preview_dir / f"{seq_name}.png")
            if "box_mask" in target_types:
                add_box_mask_preview(frames[0], first, config).save(preview_dir / f"{seq_name}_box_mask_overlay.png")

    write_csv(straight_root / "keyframes_manifest.csv", manifest_rows)


def append_straight_sequence_records(
    config: dict[str, Any],
    straight_root: Path,
    target_types: set[str],
    split: str,
    seq_name: str,
    seq_dir: Path,
    frames: list[Image.Image],
    meta: dict[str, object],
    keyframes: list[dict[str, object]],
    manifest_rows: list[dict[str, object]],
) -> None:
    for frame_info, image in zip(keyframes, frames):
        frame_name = f"frame_{int(frame_info['frame_idx']):04d}.png"
        relative_image_path = Path("sequences") / split / seq_name / frame_name
        image.save(seq_dir / frame_name)
        row = {
            "sequence_id": seq_name,
            "split": split,
            "frame_idx": frame_info["frame_idx"],
            "image": relative_image_path.as_posix(),
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
            "start_x": meta["start_x"],
            "start_y": meta["start_y"],
            "end_x": meta["end_x"],
            "end_y": meta["end_y"],
        }
        if "box_mask" in target_types:
            row["box_mask"] = save_box_mask_target(
                dataset_root=straight_root,
                relative_image_path=relative_image_path,
                metadata=frame_info,
                config=config,
            )
        manifest_rows.append(row)

    with open(seq_dir / "keyframes.json", "w", encoding="utf-8") as f:
        json.dump({**meta, "sequence_id": seq_name, "split": split, "keyframes": keyframes}, f, ensure_ascii=True, indent=2)


def generate_straight_dataset_isolated(
    config: dict[str, Any],
    output_root: Path,
    rng: random.Random,
    used_keys: set[tuple[int, int, int]],
    used_coords: set[tuple[int, int]],
    train_points_by_rotation: dict[int, list[tuple[int, int]]],
) -> None:
    straight_root = output_root / "straight"
    sequences_root = straight_root / "sequences"
    preview_dir = straight_root / "debug_preview"
    target_types = target_types_enabled(config)
    reset_dir(sequences_root)
    reset_dir(preview_dir)
    if "box_mask" in target_types:
        reset_dir(straight_root / "targets" / "box_mask")

    sequence_count = int(config["sequence_count"])
    train_count = int(round(sequence_count * float(config["train_ratio"])))
    min_distance_px = float(config.get("straight_split_min_train_distance_px", config.get("split_min_train_distance_px", 0.0)))
    preview_indices = set(range(min(sequence_count, int(config["debug_preview_count"]))))

    manifest_rows: list[dict[str, object]] = []
    params = build_shuffled_straight_params(config, rng)
    train_lookup_by_rotation: dict[int, set[tuple[int, int]]] = {}
    for rotation_deg, points in train_points_by_rotation.items():
        train_lookup_by_rotation[int(rotation_deg) % 360] = set(points)

    selected_val: list[tuple[dict[str, object], list[dict[str, object]], list[tuple[int, int, int]]]] = []
    selected_train: list[tuple[dict[str, object], list[dict[str, object]], list[tuple[int, int, int]]]] = []
    temp_used_keys = set(used_keys)
    temp_used_coords = set(used_coords)
    val_lookup_by_rotation: dict[int, set[tuple[int, int]]] = {}

    for route, lateral, trim0, trim1 in params:
        if len(selected_val) >= sequence_count - train_count:
            break
        meta, keyframes, frame_keys = build_straight_candidate_metadata(config, route, lateral, trim0, trim1)
        if len(set(frame_keys)) != int(config["frames_per_sequence"]) or any(key in temp_used_keys for key in frame_keys):
            continue
        frame_coords = [coordinate_key(key) for key in frame_keys]
        if use_unique_coordinates(config) and (len(set(frame_coords)) != int(config["frames_per_sequence"]) or any(coord in temp_used_coords for coord in frame_coords)):
            continue
        if not all_far_from_train_lookup(frame_keys, train_lookup_by_rotation, min_distance_px):
            continue
        selected_val.append((meta, keyframes, frame_keys))
        temp_used_keys.update(frame_keys)
        temp_used_coords.update(frame_coords)
        for key in frame_keys:
            add_train_lookup_key(val_lookup_by_rotation, key)
    val_count = sequence_count - train_count
    if len(selected_val) < val_count:
        raise RuntimeError(f"Only selected {len(selected_val)} isolated straight val sequences, but {val_count} are required.")

    for route, lateral, trim0, trim1 in params:
        if len(selected_train) >= train_count:
            break
        meta, keyframes, frame_keys = build_straight_candidate_metadata(config, route, lateral, trim0, trim1)
        if len(set(frame_keys)) != int(config["frames_per_sequence"]) or any(key in temp_used_keys for key in frame_keys):
            continue
        frame_coords = [coordinate_key(key) for key in frame_keys]
        if use_unique_coordinates(config) and (len(set(frame_coords)) != int(config["frames_per_sequence"]) or any(coord in temp_used_coords for coord in frame_coords)):
            continue
        if not all_far_from_train_lookup(frame_keys, val_lookup_by_rotation, min_distance_px):
            continue
        selected_train.append((meta, keyframes, frame_keys))
        temp_used_keys.update(frame_keys)
        temp_used_coords.update(frame_coords)
    if len(selected_train) < train_count:
        raise RuntimeError(f"Only selected {len(selected_train)} isolated straight train sequences, but {train_count} are required.")

    used_keys.update(temp_used_keys)
    used_coords.update(temp_used_coords)
    selected_sequences = selected_train + selected_val

    for seq_idx, (meta, keyframes, frame_keys) in enumerate(selected_sequences):
        split = "train" if seq_idx < train_count else "val"
        seq_name = f"seq_{seq_idx:05d}"
        seq_dir = sequences_root / split / seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)
        if split == "train":
            for key in frame_keys:
                add_train_key(train_points_by_rotation, key)
                add_train_lookup_key(train_lookup_by_rotation, key)

        frames = render_straight_frames(config, keyframes)
        append_straight_sequence_records(config, straight_root, target_types, split, seq_name, seq_dir, frames, meta, keyframes, manifest_rows)

        if seq_idx in preview_indices and frames:
            first = keyframes[0]
            add_preview_marker(frames[0], float(first["center_x"]), float(first["center_y"])).save(preview_dir / f"{seq_name}.png")
            if "box_mask" in target_types:
                add_box_mask_preview(frames[0], first, config).save(preview_dir / f"{seq_name}_box_mask_overlay.png")

    write_csv(straight_root / "keyframes_manifest.csv", manifest_rows)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_dataset(config: dict[str, Any]) -> None:
    output_root = Path(config["output"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=True, indent=2)

    seed = int(config["seed"])
    used_keys: set[tuple[int, int, int]] = set()
    used_coords: set[tuple[int, int]] = set()
    if float(config.get("split_min_train_distance_px", 0.0)) > 0:
        train_points_by_rotation: dict[int, list[tuple[int, int]]] = {}
        generate_random_dataset_isolated(config, output_root, random.Random(seed), used_keys, used_coords, train_points_by_rotation)
        if int(config["sequence_count"]) > 0:
            generate_straight_dataset_isolated(config, output_root, random.Random(seed + 10_000), used_keys, used_coords, train_points_by_rotation)
    else:
        generate_random_dataset(config, output_root, random.Random(seed), used_keys, used_coords)
        if int(config["sequence_count"]) > 0:
            generate_straight_dataset(config, output_root, random.Random(seed + 10_000), used_keys, used_coords)
    config["unique_image_keys"] = len(used_keys)
    config["unique_coordinate_keys"] = len(used_coords)
    with open(output_root / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=True, indent=2)


def main() -> None:
    args = parse_args()
    config = merged_config(args)
    validate_config(config)
    generate_dataset(config)


if __name__ == "__main__":
    main()
