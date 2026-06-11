import random
import math
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Dict


def _rand_color(rng: random.Random, min_brightness: int = 40) -> Tuple[int, int, int]:
    while True:
        r, g, b = rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)
        if r + g + b > min_brightness * 3:
            return (r, g, b)


def _contrasting_color(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    return (0, 0, 0) if brightness > 128 else (255, 255, 255)


def generate_shapes_scene(rng: random.Random, size: int = 224) -> Tuple[Image.Image, str]:
    bg = (rng.randint(180, 255), rng.randint(180, 255), rng.randint(180, 255))
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)

    shape_count = rng.randint(2, 5)
    shape_names = []
    color_names = {
        (255, 0, 0): "red", (0, 128, 0): "green", (0, 0, 255): "blue",
        (255, 165, 0): "orange", (128, 0, 128): "purple", (255, 255, 0): "yellow",
        (0, 128, 128): "teal", (255, 20, 147): "pink",
    }
    named_colors = list(color_names.items())

    for _ in range(shape_count):
        color_rgb, color_name = rng.choice(named_colors)
        shape = rng.choice(["circle", "rectangle", "triangle"])
        x, y = rng.randint(20, size - 60), rng.randint(20, size - 60)
        w, h = rng.randint(30, 70), rng.randint(30, 70)

        if shape == "circle":
            draw.ellipse([x, y, x + w, y + w], fill=color_rgb, outline=(0, 0, 0), width=2)
        elif shape == "rectangle":
            draw.rectangle([x, y, x + w, y + h], fill=color_rgb, outline=(0, 0, 0), width=2)
        elif shape == "triangle":
            pts = [(x + w // 2, y), (x, y + h), (x + w, y + h)]
            draw.polygon(pts, fill=color_rgb, outline=(0, 0, 0))

        shape_names.append(f"a {color_name} {shape}")

    if len(shape_names) == 1:
        caption = f"An image showing {shape_names[0]}."
    elif len(shape_names) == 2:
        caption = f"An image with {shape_names[0]} and {shape_names[1]}."
    else:
        caption = f"An image containing {', '.join(shape_names[:-1])}, and {shape_names[-1]}."

    return img, caption


def generate_gradient_scene(rng: random.Random, size: int = 224) -> Tuple[Image.Image, str]:
    direction = rng.choice(["horizontal", "vertical", "diagonal"])
    c1 = _rand_color(rng)
    c2 = _rand_color(rng)

    img = Image.new("RGB", (size, size))
    pixels = img.load()
    for y in range(size):
        for x in range(size):
            if direction == "horizontal":
                t = x / (size - 1)
            elif direction == "vertical":
                t = y / (size - 1)
            else:
                t = (x + y) / (2 * (size - 1))
            r = int(c1[0] * (1 - t) + c2[0] * t)
            g = int(c1[1] * (1 - t) + c2[1] * t)
            b = int(c1[2] * (1 - t) + c2[2] * t)
            pixels[x, y] = (r, g, b)

    caption = f"A smooth {direction} gradient blending from RGB{c1} to RGB{c2}."
    return img, caption


def generate_grid_scene(rng: random.Random, size: int = 224) -> Tuple[Image.Image, str]:
    cols = rng.randint(2, 5)
    rows = rng.randint(2, 5)
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    cell_w = size // cols
    cell_h = size // rows
    colors_used = set()

    for row in range(rows):
        for col in range(cols):
            color = _rand_color(rng)
            colors_used.add(color)
            x0, y0 = col * cell_w, row * cell_h
            x1, y1 = x0 + cell_w, y0 + cell_h
            draw.rectangle([x0, y0, x1, y1], fill=color)

    caption = f"A {cols}x{rows} grid of colored rectangles with {len(colors_used)} distinct colors."
    return img, caption


def generate_concentric_scene(rng: random.Random, size: int = 224) -> Tuple[Image.Image, str]:
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    num_rings = rng.randint(3, 7)
    ring_colors = [_rand_color(rng) for _ in range(num_rings)]

    for i in range(num_rings, 0, -1):
        r = int((i / num_rings) * (size // 2))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ring_colors[i - 1])

    caption = f"A set of {num_rings} concentric circles forming a bullseye pattern."
    return img, caption


def generate_bar_chart_scene(rng: random.Random, size: int = 224) -> Tuple[Image.Image, str]:
    img = Image.new("RGB", (size, size), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    num_bars = rng.randint(3, 7)
    bar_values = [rng.randint(20, 100) for _ in range(num_bars)]
    bar_colors = [_rand_color(rng) for _ in range(num_bars)]

    margin = 20
    chart_w = size - 2 * margin
    chart_h = size - 2 * margin
    bar_w = chart_w // num_bars - 4
    max_val = max(bar_values)

    draw.rectangle([margin, margin, size - margin, size - margin], fill=(255, 255, 255), outline=(0, 0, 0))

    for i, (val, color) in enumerate(zip(bar_values, bar_colors)):
        bar_h = int((val / max_val) * (chart_h - 10))
        x0 = margin + i * (bar_w + 4) + 4
        y0 = size - margin - bar_h
        x1 = x0 + bar_w
        y1 = size - margin
        draw.rectangle([x0, y0, x1, y1], fill=color, outline=(0, 0, 0))

    caption = f"A bar chart with {num_bars} bars showing values {bar_values}."
    return img, caption


def generate_checkerboard_scene(rng: random.Random, size: int = 224) -> Tuple[Image.Image, str]:
    n = rng.randint(4, 10)
    c1 = _rand_color(rng)
    c2 = _rand_color(rng)
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    cell = size // n

    for row in range(n):
        for col in range(n):
            color = c1 if (row + col) % 2 == 0 else c2
            draw.rectangle([col * cell, row * cell, (col + 1) * cell, (row + 1) * cell], fill=color)

    caption = f"A {n}x{n} checkerboard pattern alternating two colors."
    return img, caption


def generate_star_scene(rng: random.Random, size: int = 224) -> Tuple[Image.Image, str]:
    bg = _rand_color(rng, min_brightness=10)
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    num_stars = rng.randint(5, 20)

    for _ in range(num_stars):
        cx = rng.randint(10, size - 10)
        cy = rng.randint(10, size - 10)
        outer_r = rng.randint(8, 20)
        inner_r = outer_r // 2
        points = 5
        star_pts = []
        for i in range(points * 2):
            angle = math.pi / points * i - math.pi / 2
            r = outer_r if i % 2 == 0 else inner_r
            star_pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        star_color = _rand_color(rng)
        draw.polygon(star_pts, fill=star_color, outline=(255, 255, 255))

    caption = f"A night-sky scene with {num_stars} stars on a dark background."
    return img, caption


def generate_landscape_scene(rng: random.Random, size: int = 224) -> Tuple[Image.Image, str]:
    sky_colors = [(135, 206, 235), (70, 130, 180), (255, 165, 100), (100, 100, 180)]
    ground_colors = [(34, 139, 34), (139, 90, 43), (200, 180, 140), (80, 120, 80)]

    sky_color = rng.choice(sky_colors)
    ground_color = rng.choice(ground_colors)
    horizon = rng.randint(size // 3, 2 * size // 3)

    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size, horizon], fill=sky_color)
    draw.rectangle([0, horizon, size, size], fill=ground_color)

    sun_x, sun_y = rng.randint(30, size - 30), rng.randint(10, horizon - 20)
    sun_r = rng.randint(15, 35)
    draw.ellipse([sun_x - sun_r, sun_y - sun_r, sun_x + sun_r, sun_y + sun_r], fill=(255, 220, 50))

    num_hills = rng.randint(2, 4)
    for _ in range(num_hills):
        hx = rng.randint(0, size)
        hr = rng.randint(40, 100)
        hill_color = (
            max(0, ground_color[0] - rng.randint(10, 40)),
            min(255, ground_color[1] + rng.randint(0, 20)),
            max(0, ground_color[2] - rng.randint(10, 30)),
        )
        draw.ellipse([hx - hr, horizon - hr // 2, hx + hr, horizon + hr // 2], fill=hill_color)

    sky_desc = {(135, 206, 235): "blue", (70, 130, 180): "stormy", (255, 165, 100): "sunset", (100, 100, 180): "dusk"}
    ground_desc = {(34, 139, 34): "grassy", (139, 90, 43): "earthy", (200, 180, 140): "sandy", (80, 120, 80): "mossy"}
    caption = f"A landscape scene with a {sky_desc[sky_color]} sky and {ground_desc[ground_color]} ground, with a sun and {num_hills} rolling hills."
    return img, caption


SCENE_GENERATORS = [
    (generate_shapes_scene, "What shapes and colors do you see?"),
    (generate_gradient_scene, "Describe the color pattern in this image."),
    (generate_grid_scene, "What is shown in this image?"),
    (generate_concentric_scene, "Describe the pattern you see."),
    (generate_bar_chart_scene, "What does this chart show?"),
    (generate_checkerboard_scene, "Describe the pattern in this image."),
    (generate_star_scene, "What is depicted in this image?"),
    (generate_landscape_scene, "Describe this scene."),
]


def generate_sample(idx: int, size: int = 224) -> Tuple[Image.Image, str, str]:
    rng = random.Random(idx * 7919 + 42)
    gen_fn, question = SCENE_GENERATORS[idx % len(SCENE_GENERATORS)]
    image, caption = gen_fn(rng, size)
    return image, caption, question
