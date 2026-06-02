import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


CAR_COLORS = {
    "red": (198, 59, 64),
    "blue": (55, 96, 187),
    "green": (73, 132, 89),
    "yellow": (208, 171, 51),
    "white": (232, 232, 228),
    "black": (42, 42, 44),
    "silver": (156, 162, 172),
    "orange": (207, 114, 52),
}

ROAD_PALETTE = {
    "asphalt": (68, 70, 73),
    "asphalt_dark": (57, 59, 62),
    "line_white": (240, 238, 220),
    "line_yellow": (233, 193, 79),
    "curb": (160, 160, 156),
    "sidewalk": (192, 188, 180),
    "grass": (92, 128, 84),
}

BACKGROUND_STYLES = (
    "urban_grid",
    "offset_grid",
    "avenue_blocks",
    "mixed_network",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic top-down vehicle localization dataset."
    )
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--safety-margin", type=int, default=32)
    parser.add_argument("--debug-preview-count", type=int, default=10)
    return parser.parse_args()


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def clear_pngs(directory: Path) -> None:
    if not directory.exists():
        return
    for image_path in directory.glob("*.png"):
        image_path.unlink()


def draw_asphalt_texture(image: Image.Image, rng: random.Random) -> None:
    np_rng = np.random.default_rng(rng.randint(0, 1_000_000))
    array = np.array(image, dtype=np.int16)
    noise = np_rng.integers(-10, 11, size=array.shape, endpoint=False)
    textured = np.clip(array + noise, 0, 255).astype(np.uint8)
    image.paste(Image.fromarray(textured))


def draw_crosswalk(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, horizontal: bool) -> None:
    if horizontal:
        gap = max(6, h // 5)
        stripe_count = max(5, round((w + gap) / 48))
        stripe_w = max(12, (w - gap * (stripe_count - 1)) // stripe_count)
        total_w = stripe_count * stripe_w + (stripe_count - 1) * gap
        start_x = x + max(0, (w - total_w) // 2)
        for i in range(stripe_count):
            stripe_x = start_x + i * (stripe_w + gap)
            draw.rectangle((stripe_x, y, stripe_x + stripe_w, y + h), fill=ROAD_PALETTE["line_white"])
    else:
        gap = max(6, w // 5)
        stripe_count = max(5, round((h + gap) / 48))
        stripe_h = max(12, (h - gap * (stripe_count - 1)) // stripe_count)
        total_h = stripe_count * stripe_h + (stripe_count - 1) * gap
        start_y = y + max(0, (h - total_h) // 2)
        for i in range(stripe_count):
            stripe_y = start_y + i * (stripe_h + gap)
            draw.rectangle((x, stripe_y, x + w, stripe_y + stripe_h), fill=ROAD_PALETTE["line_white"])


def add_region(
    regions: list[dict[str, object]],
    bbox: tuple[int, int, int, int],
    headings: tuple[int, ...],
    jitter_deg: int,
) -> None:
    x0, y0, x1, y1 = bbox
    if x1 - x0 >= 72 and y1 - y0 >= 72:
        regions.append({"bbox": bbox, "headings": headings, "jitter_deg": jitter_deg})


def texture_patch(image: Image.Image, bbox: tuple[int, int, int, int], rng: random.Random) -> None:
    patch = image.crop(bbox)
    draw_asphalt_texture(patch, rng)
    image.paste(patch, bbox[:2])


def draw_road_surface(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    bbox: tuple[int, int, int, int],
    orientation: str,
    rng: random.Random,
) -> None:
    x0, y0, x1, y1 = bbox
    draw.rectangle(bbox, fill=ROAD_PALETTE["asphalt_dark"])
    texture_patch(image, bbox, rng)

    curb_w = 7
    side_line_w = 4
    if orientation == "vertical":
        draw.rectangle((x0 - curb_w, y0, x0, y1), fill=ROAD_PALETTE["curb"])
        draw.rectangle((x1, y0, x1 + curb_w, y1), fill=ROAD_PALETTE["curb"])
        draw.rectangle((x0 + 16, y0, x0 + 16 + side_line_w, y1), fill=ROAD_PALETTE["line_white"])
        draw.rectangle((x1 - 16 - side_line_w, y0, x1 - 16, y1), fill=ROAD_PALETTE["line_white"])
        dash_w = 8
        dash_h = rng.randint(34, 48)
        gap = rng.randint(24, 34)
        lane_x = (x0 + x1) // 2
        current_y = 18
        while current_y < y1:
            draw.rectangle(
                (lane_x - dash_w // 2, current_y, lane_x + dash_w // 2, current_y + dash_h),
                fill=ROAD_PALETTE["line_yellow"],
            )
            current_y += dash_h + gap
    else:
        draw.rectangle((x0, y0 - curb_w, x1, y0), fill=ROAD_PALETTE["curb"])
        draw.rectangle((x0, y1, x1, y1 + curb_w), fill=ROAD_PALETTE["curb"])
        draw.rectangle((x0, y0 + 16, x1, y0 + 16 + side_line_w), fill=ROAD_PALETTE["line_white"])
        draw.rectangle((x0, y1 - 16 - side_line_w, x1, y1 - 16), fill=ROAD_PALETTE["line_white"])
        dash_h = 8
        dash_w = rng.randint(34, 48)
        gap = rng.randint(24, 34)
        lane_y = (y0 + y1) // 2
        current_x = 18
        while current_x < x1:
            draw.rectangle(
                (current_x, lane_y - dash_h // 2, current_x + dash_w, lane_y + dash_h // 2),
                fill=ROAD_PALETTE["line_yellow"],
            )
            current_x += dash_w + gap


def add_segmented_regions(
    regions: list[dict[str, object]],
    road_bbox: tuple[int, int, int, int],
    orientation: str,
    blocked_intervals: list[tuple[int, int]],
    jitter_deg: int,
) -> None:
    x0, y0, x1, y1 = road_bbox
    lane_margin = 34
    edge_margin = 20

    if orientation == "vertical":
        base_x0 = x0 + lane_margin
        base_x1 = x1 - lane_margin
        if base_x1 - base_x0 < 96:
            base_x0 = x0 + 24
            base_x1 = x1 - 24
        intervals = [(max(y0 + edge_margin, a), min(y1 - edge_margin, b)) for a, b in blocked_intervals]
        intervals = [(a, b) for a, b in intervals if b > a]
        intervals.sort()
        cursor = y0 + edge_margin
        for start, end in intervals:
            add_region(regions, (base_x0, cursor, base_x1, start), (0, 180), jitter_deg)
            cursor = max(cursor, end)
        add_region(regions, (base_x0, cursor, base_x1, y1 - edge_margin), (0, 180), jitter_deg)
    else:
        base_y0 = y0 + lane_margin
        base_y1 = y1 - lane_margin
        if base_y1 - base_y0 < 96:
            base_y0 = y0 + 24
            base_y1 = y1 - 24
        intervals = [(max(x0 + edge_margin, a), min(x1 - edge_margin, b)) for a, b in blocked_intervals]
        intervals = [(a, b) for a, b in intervals if b > a]
        intervals.sort()
        cursor = x0 + edge_margin
        for start, end in intervals:
            add_region(regions, (cursor, base_y0, start, base_y1), (90, 270), jitter_deg)
            cursor = max(cursor, end)
        add_region(regions, (cursor, base_y0, x1 - edge_margin, base_y1), (90, 270), jitter_deg)


def draw_intersection_crosswalks(
    draw: ImageDraw.ImageDraw,
    vertical_bbox: tuple[int, int, int, int],
    horizontal_bbox: tuple[int, int, int, int],
) -> None:
    vx0, vy0, vx1, vy1 = vertical_bbox
    hx0, hy0, hx1, hy1 = horizontal_bbox
    cross_x0 = max(vx0, hx0)
    cross_y0 = max(vy0, hy0)
    cross_x1 = min(vx1, hx1)
    cross_y1 = min(vy1, hy1)
    if cross_x1 <= cross_x0 or cross_y1 <= cross_y0:
        return

    cw_w = max(72, cross_x1 - cross_x0 - 24)
    cw_h = max(72, cross_y1 - cross_y0 - 24)
    draw_crosswalk(draw, cross_x0 + 12, cross_y0 - 52, cw_w, 34, horizontal=True)
    draw_crosswalk(draw, cross_x0 + 12, cross_y1 + 18, cw_w, 34, horizontal=True)
    draw_crosswalk(draw, cross_x0 - 52, cross_y0 + 12, 34, cw_h, horizontal=False)
    draw_crosswalk(draw, cross_x1 + 18, cross_y0 + 12, 34, cw_h, horizontal=False)


def make_background(image_size: int, rng: random.Random) -> tuple[Image.Image, str, list[dict[str, object]]]:
    style = rng.choice(BACKGROUND_STYLES)
    image = Image.new("RGB", (image_size, image_size), ROAD_PALETTE["sidewalk"])
    draw = ImageDraw.Draw(image)
    regions: list[dict[str, object]] = []

    outer_green = rng.randint(22, 40)
    draw.rectangle((0, 0, outer_green, image_size), fill=ROAD_PALETTE["grass"])
    draw.rectangle((image_size - outer_green, 0, image_size, image_size), fill=ROAD_PALETTE["grass"])
    draw.rectangle((0, 0, image_size, outer_green), fill=ROAD_PALETTE["grass"])
    draw.rectangle((0, image_size - outer_green, image_size, image_size), fill=ROAD_PALETTE["grass"])

    if style == "urban_grid":
        vertical_centers = [int(image_size * 0.32), int(image_size * 0.68)]
        horizontal_centers = [int(image_size * 0.33), int(image_size * 0.69)]
    elif style == "offset_grid":
        vertical_centers = [int(image_size * 0.24), int(image_size * 0.56), int(image_size * 0.82)]
        horizontal_centers = [int(image_size * 0.28), int(image_size * 0.72)]
    elif style == "avenue_blocks":
        vertical_centers = [int(image_size * 0.42), int(image_size * 0.78)]
        horizontal_centers = [int(image_size * 0.22), int(image_size * 0.52), int(image_size * 0.82)]
    else:
        vertical_centers = [int(image_size * 0.28), int(image_size * 0.74)]
        horizontal_centers = [int(image_size * 0.38), int(image_size * 0.64)]
        vertical_centers.append(int(image_size * 0.52))

    vertical_roads = []
    horizontal_roads = []
    for center in vertical_centers:
        road_w = rng.randint(110, 150)
        x0 = max(outer_green + 18, center - road_w // 2)
        x1 = min(image_size - outer_green - 18, center + road_w // 2)
        vertical_roads.append((x0, 0, x1, image_size))

    for center in horizontal_centers:
        road_h = rng.randint(110, 150)
        y0 = max(outer_green + 18, center - road_h // 2)
        y1 = min(image_size - outer_green - 18, center + road_h // 2)
        horizontal_roads.append((0, y0, image_size, y1))

    for bbox in vertical_roads:
        draw_road_surface(image, draw, bbox, "vertical", rng)
    for bbox in horizontal_roads:
        draw_road_surface(image, draw, bbox, "horizontal", rng)

    blocked_vertical: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(vertical_roads))}
    blocked_horizontal: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(horizontal_roads))}
    for vi, v_bbox in enumerate(vertical_roads):
        for hi, h_bbox in enumerate(horizontal_roads):
            cross_x0 = max(v_bbox[0], h_bbox[0])
            cross_y0 = max(v_bbox[1], h_bbox[1])
            cross_x1 = min(v_bbox[2], h_bbox[2])
            cross_y1 = min(v_bbox[3], h_bbox[3])
            if cross_x1 <= cross_x0 or cross_y1 <= cross_y0:
                continue
            if rng.random() < 0.75:
                draw_intersection_crosswalks(draw, v_bbox, h_bbox)
            blocked_vertical[vi].append((cross_y0 - 72, cross_y1 + 72))
            blocked_horizontal[hi].append((cross_x0 - 72, cross_x1 + 72))

    for road_index, bbox in enumerate(vertical_roads):
        add_segmented_regions(regions, bbox, "vertical", blocked_vertical[road_index], jitter_deg=8)
    for road_index, bbox in enumerate(horizontal_roads):
        add_segmented_regions(regions, bbox, "horizontal", blocked_horizontal[road_index], jitter_deg=8)

    block_outlines = []
    x_edges = [outer_green] + [bbox[0] for bbox in vertical_roads] + [bbox[2] for bbox in vertical_roads] + [image_size - outer_green]
    y_edges = [outer_green] + [bbox[1] for bbox in horizontal_roads] + [bbox[3] for bbox in horizontal_roads] + [image_size - outer_green]
    x_edges = sorted(set(x_edges))
    y_edges = sorted(set(y_edges))
    for i in range(len(x_edges) - 1):
        for j in range(len(y_edges) - 1):
            x0 = x_edges[i]
            x1 = x_edges[i + 1]
            y0 = y_edges[j]
            y1 = y_edges[j + 1]
            if x1 - x0 < 60 or y1 - y0 < 60:
                continue
            inset = rng.randint(10, 18)
            block_box = (x0 + inset, y0 + inset, x1 - inset, y1 - inset)
            if block_box[2] - block_box[0] > 36 and block_box[3] - block_box[1] > 36:
                block_outlines.append(block_box)

    for block_box in block_outlines[: min(len(block_outlines), 12)]:
        if rng.random() < 0.55:
            draw.rectangle(block_box, outline=ROAD_PALETTE["line_white"], width=3)

    return image, style, regions


def draw_rounded_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    radius: int,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int] | None = None,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def make_vehicle_sprite(scale: float, color_name: str) -> tuple[Image.Image, Image.Image]:
    body_w = int(round(52 * scale))
    body_h = int(round(100 * scale))
    canvas_w = body_w + int(round(36 * scale))
    canvas_h = body_h + int(round(36 * scale))
    image = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    vehicle_mask = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(vehicle_mask)

    body_x0 = (canvas_w - body_w) // 2
    body_y0 = (canvas_h - body_h) // 2
    body_x1 = body_x0 + body_w
    body_y1 = body_y0 + body_h
    radius = max(8, int(round(14 * scale)))
    body_color = CAR_COLORS[color_name]
    outline = tuple(clamp(channel - 28, 0, 255) for channel in body_color)
    shadow_offset = int(round(5 * scale))

    shadow = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    draw_rounded_rectangle(
        shadow_draw,
        (body_x0 + shadow_offset, body_y0 + shadow_offset, body_x1 + shadow_offset, body_y1 + shadow_offset),
        radius=radius,
        fill=(0, 0, 0, 120),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(3, int(round(6 * scale)))))
    image = Image.alpha_composite(image, shadow)
    draw = ImageDraw.Draw(image)

    draw_rounded_rectangle(
        draw,
        (body_x0, body_y0, body_x1, body_y1),
        radius=radius,
        fill=(*body_color, 255),
        outline=(*outline, 255),
        width=max(2, int(round(3 * scale))),
    )
    draw_rounded_rectangle(
        mask_draw,
        (body_x0, body_y0, body_x1, body_y1),
        radius=radius,
        fill=(255, 255, 255, 255),
    )

    roof_margin_x = int(round(10 * scale))
    roof_margin_y = int(round(18 * scale))
    roof_box = (
        body_x0 + roof_margin_x,
        body_y0 + roof_margin_y,
        body_x1 - roof_margin_x,
        body_y1 - roof_margin_y,
    )
    roof_color = tuple(clamp(channel + 24, 0, 255) for channel in body_color)
    draw_rounded_rectangle(
        draw,
        roof_box,
        radius=max(6, int(round(10 * scale))),
        fill=(*roof_color, 240),
        outline=(255, 255, 255, 90),
        width=max(1, int(round(2 * scale))),
    )

    glass_margin = int(round(5 * scale))
    windshield = (
        roof_box[0] + glass_margin,
        roof_box[1] + glass_margin,
        roof_box[2] - glass_margin,
        (roof_box[1] + roof_box[3]) // 2 - int(round(4 * scale)),
    )
    rear_window = (
        roof_box[0] + glass_margin,
        (roof_box[1] + roof_box[3]) // 2 + int(round(4 * scale)),
        roof_box[2] - glass_margin,
        roof_box[3] - glass_margin,
    )
    glass_color = (86, 126, 160, 210)
    draw_rounded_rectangle(draw, windshield, radius=max(5, int(round(8 * scale))), fill=glass_color)
    draw_rounded_rectangle(draw, rear_window, radius=max(5, int(round(8 * scale))), fill=glass_color)

    center_y = (roof_box[1] + roof_box[3]) // 2
    draw.line(
        (roof_box[0] + 4, center_y, roof_box[2] - 4, center_y),
        fill=(250, 250, 250, 150),
        width=max(1, int(round(2 * scale))),
    )

    wheel_w = max(7, int(round(9 * scale)))
    wheel_h = max(15, int(round(18 * scale)))
    wheel_x_offsets = (body_x0 - wheel_w // 2, body_x1 - wheel_w // 2)
    wheel_y_offsets = (
        body_y0 + int(round(14 * scale)),
        body_y1 - int(round(14 * scale)) - wheel_h,
    )
    for wheel_x in wheel_x_offsets:
        for wheel_y in wheel_y_offsets:
            draw.rounded_rectangle(
                (wheel_x, wheel_y, wheel_x + wheel_w, wheel_y + wheel_h),
                radius=max(3, int(round(4 * scale))),
                fill=(28, 28, 30, 255),
            )
            mask_draw.rounded_rectangle(
                (wheel_x, wheel_y, wheel_x + wheel_w, wheel_y + wheel_h),
                radius=max(3, int(round(4 * scale))),
                fill=(255, 255, 255, 255),
            )

    front_marker_y = body_y0 + int(round(8 * scale))
    rear_marker_y = body_y1 - int(round(8 * scale))
    marker_inset = int(round(12 * scale))
    draw.ellipse(
        (body_x0 + marker_inset, front_marker_y, body_x0 + marker_inset + 10, front_marker_y + 6),
        fill=(255, 235, 170, 220),
    )
    draw.ellipse(
        (body_x1 - marker_inset - 10, front_marker_y, body_x1 - marker_inset, front_marker_y + 6),
        fill=(255, 235, 170, 220),
    )
    draw.ellipse(
        (body_x0 + marker_inset, rear_marker_y, body_x0 + marker_inset + 10, rear_marker_y + 6),
        fill=(215, 82, 82, 220),
    )
    draw.ellipse(
        (body_x1 - marker_inset - 10, rear_marker_y, body_x1 - marker_inset, rear_marker_y + 6),
        fill=(215, 82, 82, 220),
    )

    return image, vehicle_mask


def rotate_vehicle(
    sprite: Image.Image, vehicle_mask: Image.Image, rotation_deg: int
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    rotated_sprite = sprite.rotate(rotation_deg, resample=Image.Resampling.BICUBIC, expand=True)
    rotated_mask = vehicle_mask.rotate(rotation_deg, resample=Image.Resampling.BICUBIC, expand=True)
    sprite_bbox = rotated_sprite.getchannel("A").getbbox()
    if sprite_bbox is None:
        raise RuntimeError("Vehicle sprite became fully transparent after rotation.")

    cropped_sprite = rotated_sprite.crop(sprite_bbox)
    cropped_mask = rotated_mask.crop(sprite_bbox)
    vehicle_bbox = cropped_mask.getchannel("A").getbbox()
    if vehicle_bbox is None:
        raise RuntimeError("Vehicle mask became fully transparent after rotation.")
    return cropped_sprite, vehicle_bbox


def choose_vehicle_position(
    image_size: int,
    safety_margin: int,
    vehicle_bbox: tuple[int, int, int, int],
    sprite_size: tuple[int, int],
    allowed_region: tuple[int, int, int, int],
    rng: random.Random,
) -> dict[str, float | int]:
    vehicle_width = vehicle_bbox[2] - vehicle_bbox[0]
    vehicle_height = vehicle_bbox[3] - vehicle_bbox[1]
    region_x0, region_y0, region_x1, region_y1 = allowed_region
    min_x = max(safety_margin, vehicle_bbox[0], region_x0)
    min_y = max(safety_margin, vehicle_bbox[1], region_y0)
    max_x = min(
        image_size - safety_margin - vehicle_width,
        image_size - (sprite_size[0] - vehicle_bbox[0]),
        region_x1 - vehicle_width,
    )
    max_y = min(
        image_size - safety_margin - vehicle_height,
        image_size - (sprite_size[1] - vehicle_bbox[1]),
        region_y1 - vehicle_height,
    )
    if max_x < min_x or max_y < min_y:
        raise ValueError("Vehicle is too large to fit within the requested image size.")

    bbox_x = rng.randint(min_x, max_x)
    bbox_y = rng.randint(min_y, max_y)
    paste_x = bbox_x - vehicle_bbox[0]
    paste_y = bbox_y - vehicle_bbox[1]

    if paste_x < 0 or paste_y < 0:
        raise ValueError("Sprite padding exceeded the safety margin budget.")
    if paste_x + sprite_size[0] > image_size or paste_y + sprite_size[1] > image_size:
        raise ValueError("Rotated sprite would fall outside the canvas bounds.")

    center_x = bbox_x + vehicle_width / 2.0
    center_y = bbox_y + vehicle_height / 2.0
    return {
        "paste_x": paste_x,
        "paste_y": paste_y,
        "center_x": round(center_x, 2),
        "center_y": round(center_y, 2),
        "car_width": vehicle_width,
        "car_height": vehicle_height,
    }


def sample_rotation(region: dict[str, object], rng: random.Random) -> int:
    headings = tuple(int(angle) for angle in region["headings"])
    jitter_deg = int(region["jitter_deg"])
    base_heading = rng.choice(headings)
    return (base_heading + rng.randint(-jitter_deg, jitter_deg)) % 360


def place_vehicle_in_scene(
    image_size: int,
    safety_margin: int,
    sprite: Image.Image,
    vehicle_mask: Image.Image,
    regions: list[dict[str, object]],
    rng: random.Random,
) -> tuple[Image.Image, dict[str, float | int], int]:
    for _ in range(120):
        region = rng.choice(regions)
        rotation_deg = sample_rotation(region, rng)
        rotated_sprite, vehicle_bbox = rotate_vehicle(sprite, vehicle_mask, rotation_deg)
        try:
            placement = choose_vehicle_position(
                image_size=image_size,
                safety_margin=safety_margin,
                vehicle_bbox=vehicle_bbox,
                sprite_size=rotated_sprite.size,
                allowed_region=tuple(region["bbox"]),
                rng=rng,
            )
            return rotated_sprite, placement, rotation_deg
        except ValueError:
            continue
    raise RuntimeError("Failed to place the vehicle within a valid drivable region.")


def add_preview_marker(image: Image.Image, center_x: float, center_y: float) -> Image.Image:
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    x = int(round(center_x))
    y = int(round(center_y))
    size = 10
    draw.line((x - size, y, x + size, y), fill=(255, 48, 48), width=2)
    draw.line((x, y - size, x, y + size), fill=(255, 48, 48), width=2)
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(255, 255, 255), width=2)
    return preview


def apply_render_variation(image: Image.Image, rng: random.Random) -> Image.Image:
    result = image
    enhancers = (
        (ImageEnhance.Brightness, rng.uniform(0.9, 1.08)),
        (ImageEnhance.Contrast, rng.uniform(0.88, 1.12)),
        (ImageEnhance.Color, rng.uniform(0.85, 1.15)),
    )
    for enhancer_cls, factor in enhancers:
        result = enhancer_cls(result).enhance(factor)

    if rng.random() < 0.7:
        result = result.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.8)))

    if rng.random() < 0.45:
        resized_side = rng.choice((560, 576, 592, 608))
        result = result.resize((resized_side, resized_side), resample=Image.Resampling.BILINEAR)
        result = result.resize((image.size[0], image.size[1]), resample=Image.Resampling.BICUBIC)

    np_rng = np.random.default_rng(rng.randint(0, 1_000_000))
    array = np.array(result, dtype=np.int16)
    noise = np_rng.integers(-8, 9, size=array.shape, endpoint=False)
    array = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def build_sample(
    image_size: int,
    safety_margin: int,
    rng: random.Random,
) -> tuple[Image.Image, dict[str, object]]:
    for _ in range(120):
        background, background_style, regions = make_background(image_size, rng)
        if not regions:
            continue
        color_name = rng.choice(tuple(CAR_COLORS.keys()))
        scale = round(rng.uniform(0.68, 1.0), 3)
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
        except RuntimeError:
            continue
    raise RuntimeError("Failed to build a valid sample after multiple retries.")


def validate_annotations(
    records: list[dict[str, object]],
    image_size: int,
    train_count: int,
    total_count: int,
    allow_out_of_bounds_center: bool = False,
) -> None:
    if len(records) != total_count:
        raise AssertionError("Annotation count does not match the number of generated images.")

    train_rows = sum(1 for record in records if record["split"] == "train")
    val_rows = sum(1 for record in records if record["split"] == "val")
    if train_rows != train_count or val_rows != total_count - train_count:
        raise AssertionError("Train/val split counts do not match the requested ratio.")

    for record in records:
        x = float(record["center_x"])
        y = float(record["center_y"])
        if not allow_out_of_bounds_center and not (0 <= x < image_size and 0 <= y < image_size):
            raise AssertionError("Found a center point that falls outside the image bounds.")


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
        image, metadata = build_sample(
            image_size=args.image_size,
            safety_margin=args.safety_margin,
            rng=rng,
        )
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
    annotations = pd.DataFrame.from_records(records)
    annotations.to_csv(output_root / "annotations.csv", index=False)


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
