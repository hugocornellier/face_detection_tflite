#!/usr/bin/env python3
"""Compose a styled face-embedding demo image from real app output.

Scores and bounding boxes come from integration_test/embedding_demo_test.dart
run against the actual MobileFaceNet pipeline on macOS.
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SAMPLES = "assets/samples/demo"

# (path, bbox left, top, w, h) straight from the app's detector output.
DAY3 = (f"{SAMPLES}/hugo_day3.jpg", 217.3, 189.3, 206.6, 206.6)
DAY4 = (f"{SAMPLES}/hugo_day4.jpg", 238.9, 183.4, 171.9, 171.9)
MAN = (f"{SAMPLES}/person_b1.jpg", 366.0, 191.7, 505.7, 504.5)
WOMAN = (f"{SAMPLES}/person_b2.jpg", 249.7, 270.6, 258.1, 258.1)

SAME_SCORE = 0.8991
DIFF_SCORE = 0.2544
THRESHOLD = 0.60

# Palette (dark, modern).
BG_TOP = (16, 18, 27)
BG_BOT = (24, 27, 38)
CARD = (30, 34, 47)
CARD_EDGE = (52, 58, 78)
TEXT = (236, 239, 247)
MUTED = (138, 146, 168)
GREEN = (52, 211, 153)
GREEN_DIM = (16, 60, 52)
RED = (248, 113, 113)
RED_DIM = (66, 26, 32)

FONT_PATH = "/System/Library/Fonts/SFNS.ttf"
BOLD_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
REG_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"


def font(size, bold=False):
    try:
        f = ImageFont.truetype(FONT_PATH, size)
        if bold:
            try:
                f.set_variation_by_name("Bold")
            except Exception:
                f = ImageFont.truetype(BOLD_PATH, size)
        return f
    except Exception:
        return ImageFont.truetype(BOLD_PATH if bold else REG_PATH, size)


def face_thumb(spec, size, pad=0.55):
    """Crop a square face thumbnail around the detected bbox with padding."""
    path, x, y, w, h = spec
    img = Image.open(path).convert("RGB")
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1 + pad)
    half = side / 2
    box = [cx - half, cy - half, cx + half, cy + half]
    # keep inside image
    if box[0] < 0:
        box[2] -= box[0]; box[0] = 0
    if box[1] < 0:
        box[3] -= box[1]; box[1] = 0
    if box[2] > img.width:
        box[0] -= box[2] - img.width; box[2] = img.width
    if box[3] > img.height:
        box[1] -= box[3] - img.height; box[3] = img.height
    box = [max(0, box[0]), max(0, box[1]),
           min(img.width, box[2]), min(img.height, box[3])]
    crop = img.crop([int(v) for v in box]).resize((size, size), Image.LANCZOS)
    return crop


def rounded(img, radius):
    mask = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, img.size[0] - 1, img.size[1] - 1],
                        radius=radius, fill=255)
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    return out


def gradient_bg(w, h):
    base = Image.new("RGB", (w, h), BG_TOP)
    top = Image.new("RGB", (w, h), BG_TOP)
    bot = Image.new("RGB", (w, h), BG_BOT)
    mask = Image.new("L", (1, h))
    for i in range(h):
        mask.putpixel((0, i), int(255 * i / h))
    mask = mask.resize((w, h))
    return Image.composite(bot, top, mask)


def draw_text_center(d, cx, y, text, fnt, fill):
    bb = d.textbbox((0, 0), text, font=fnt)
    d.text((cx - (bb[2] - bb[0]) / 2, y), text, font=fnt, fill=fill)
    return bb[3] - bb[1]


def panel(draw, base, x, y, w, h, left_spec, right_spec, score, accent,
          accent_dim, verdict, sublabel):
    # Card
    draw.rounded_rectangle([x, y, x + w, y + h], radius=28,
                           fill=CARD, outline=CARD_EDGE, width=2)

    thumb = 232
    pad = 46
    ty = y + (h - thumb) // 2 - 6
    lx = x + pad
    rx = x + w - pad - thumb

    for tx, spec in ((lx, left_spec), (rx, right_spec)):
        # subtle shadow
        shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        sd.rounded_rectangle([tx + 6, ty + 10, tx + thumb + 6, ty + thumb + 10],
                             radius=24, fill=(0, 0, 0, 120))
        shadow = shadow.filter(ImageFilter.GaussianBlur(10))
        base.alpha_composite(shadow)
        t = rounded(face_thumb(spec, thumb), 24)
        # accent ring
        ring = Image.new("RGBA", (thumb + 8, thumb + 8), (0, 0, 0, 0))
        rd = ImageDraw.Draw(ring)
        rd.rounded_rectangle([0, 0, thumb + 7, thumb + 7], radius=27,
                             outline=accent + (255,), width=3)
        base.paste(t, (tx, ty), t)
        base.alpha_composite(ring, (tx - 4, ty - 4))

    # Center column
    ccx = x + w // 2
    cd = ImageDraw.Draw(base)

    # Connector line between thumbs through center
    line_y = ty + thumb // 2
    cd.line([(lx + thumb + 6, line_y), (ccx - 90, line_y)], fill=CARD_EDGE, width=3)
    cd.line([(ccx + 90, line_y), (rx - 6, line_y)], fill=CARD_EDGE, width=3)

    # Score pill
    pill_w, pill_h = 168, 88
    px0 = ccx - pill_w // 2
    py0 = line_y - pill_h // 2
    cd.rounded_rectangle([px0, py0, px0 + pill_w, py0 + pill_h], radius=20,
                         fill=accent_dim, outline=accent + (255,), width=2)
    draw_text_center(cd, ccx, py0 + 12, f"{score:.2f}", font(46, bold=True), accent)
    draw_text_center(cd, ccx, py0 + 62, "similarity", font(17), MUTED)

    # Verdict below
    vy = ty + thumb + 20
    draw_text_center(cd, ccx, vy, verdict, font(30, bold=True), accent)


def main():
    W = 1360
    H = 956
    bg = gradient_bg(W, H).convert("RGBA")
    d = ImageDraw.Draw(bg)

    # Header
    draw_text_center(d, W // 2, 54, "Face Embedding", font(58, bold=True), TEXT)
    draw_text_center(d, W // 2, 128,
                     "On-device identity verification  ·  MobileFaceNet  ·  192-D vectors",
                     font(22), MUTED)

    px = 70
    pw = W - 2 * px
    ph = 348
    gap = 40
    py1 = 196
    py2 = py1 + ph + gap

    panel(d, bg, px, py1, pw, ph, DAY3, DAY4, SAME_SCORE,
          GREEN, GREEN_DIM, "SAME PERSON", "Hugo  ·  Day 3 vs Day 4")
    d = ImageDraw.Draw(bg)
    panel(d, bg, px, py2, pw, ph, MAN, WOMAN, DIFF_SCORE,
          RED, RED_DIM, "DIFFERENT PEOPLE", "Two distinct subjects")
    d = ImageDraw.Draw(bg)

    out = "assets/samples/demo/embedding_demo.png"
    bg.convert("RGB").save(out, quality=95)
    print("wrote", out)


if __name__ == "__main__":
    main()
