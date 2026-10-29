"""Draw the three placeholder pictures shown in the demo's empty state.

They are generated, not photographed, so they carry no licence terms and no class label:
python scripts/make_samples.py
Replace frontend/public/samples/sample-N.jpg with road photos you are allowed to use.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parents[1] / "frontend" / "public" / "samples"
WIDTH, HEIGHT = 720, 480  # 3:2, the shape the model sees (it is resized to 360 x 240)
HORIZON = int(HEIGHT * 0.38)


def scene(sky, road, edge, seed: int, streaks: int = 0, snow: bool = False) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (WIDTH, HEIGHT), sky)
    draw = ImageDraw.Draw(img)
    vanish = (WIDTH * 0.5, HORIZON)
    draw.polygon(
        [(0, HEIGHT), (WIDTH, HEIGHT), (WIDTH * 0.56, HORIZON), (WIDTH * 0.44, HORIZON)], fill=road
    )
    draw.polygon([(0, HEIGHT), (WIDTH * 0.06, HEIGHT), (WIDTH * 0.44, HORIZON)], fill=edge)
    draw.polygon([(WIDTH, HEIGHT), (WIDTH * 0.94, HEIGHT), (WIDTH * 0.56, HORIZON)], fill=edge)
    for k in range(7):  # dashed centre line receding to the horizon
        t0, t1 = (k / 7) ** 1.8, ((k + 0.55) / 7) ** 1.8
        y0, y1 = HORIZON + (HEIGHT - HORIZON) * t0, HORIZON + (HEIGHT - HORIZON) * t1
        w0, w1 = 2 + 10 * t0, 2 + 10 * t1
        draw.polygon(
            [
                (vanish[0] - w0, y0),
                (vanish[0] + w0, y0),
                (vanish[0] + w1, y1),
                (vanish[0] - w1, y1),
            ],
            fill=(235, 232, 220),
        )
    arr = np.asarray(img).astype(np.float32)
    arr += rng.normal(0, 7, arr.shape)  # surface grain
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    layer = ImageDraw.Draw(img, "RGBA")
    for _ in range(streaks):  # glossy reflections
        x = rng.uniform(WIDTH * 0.25, WIDTH * 0.75)
        layer.rectangle(
            [x, HORIZON + 20, x + rng.uniform(4, 18), HEIGHT],
            fill=(220, 235, 250, int(rng.uniform(30, 70))),
        )
    if snow:
        for _ in range(900):
            x, y = rng.uniform(0, WIDTH), rng.uniform(HORIZON, HEIGHT)
            r = rng.uniform(1, 3)
            layer.ellipse([x, y, x + r, y + r], fill=(255, 255, 255, 140))
    return img.filter(ImageFilter.GaussianBlur(0.8))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    specs = [
        scene((170, 190, 210), (74, 76, 80), (110, 112, 100), seed=1),
        scene((120, 130, 150), (58, 64, 74), (80, 86, 96), seed=2, streaks=14),
        scene((205, 210, 215), (196, 200, 204), (225, 228, 230), seed=3, snow=True),
    ]
    for i, img in enumerate(specs, start=1):
        path = OUT / f"sample-{i}.jpg"
        img.save(path, quality=85)
        print(f"wrote {path.relative_to(OUT.parents[2])} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    sys.exit(main())
