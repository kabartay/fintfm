"""Draw images/logo.png: `python3 scripts/draw_logo.py images/logo.png` (needs Pillow).

FinTFM mark: a table whose labelled rows are the context, read in one forward pass,
and a new row whose last cell is the predicted default probability."""
import sys

from PIL import Image, ImageDraw

S = 1024
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
bg, gold, white, dim, dim2 = (15, 26, 43, 255), (242, 177, 52, 255), (240, 244, 248, 255), (70, 90, 115, 255), (48, 64, 86, 255)
d.rounded_rectangle([0, 0, S - 1, S - 1], radius=200, fill=bg)
cols, rows = 3, 4
x0, y0, cw, ch, gx, gy = 168, 269, 150, 96, 30, 34
for r in range(rows):
    for c in range(cols):
        x = x0 + c * (cw + gx); y = y0 + r * (ch + gy)
        box = [x, y, x + cw, y + ch]
        last = c == cols - 1
        if r < rows - 1:                       # context rows: features dim, label column bright
            d.rounded_rectangle(box, radius=22, fill=white if last else dim)
        else:                                  # the new row: features outlined, label predicted
            if last:
                d.rounded_rectangle(box, radius=22, fill=gold)
            else:
                d.rounded_rectangle(box, radius=22, outline=dim, width=12, fill=bg)
# the single forward pass: one line gathering the context rows and delivering the prediction
bx = x0 + cols * (cw + gx) + 96
top = y0 + ch / 2
ay = y0 + (rows - 1) * (ch + gy) + ch / 2
right = x0 + cols * (cw + gx) - gx
d.line([(right + 14, top), (bx, top)], fill=gold, width=22)
d.line([(bx, top - 11), (bx, ay)], fill=gold, width=22)
d.line([(bx, ay), (right + 90, ay)], fill=gold, width=22)
d.polygon([(right + 22, ay), (right + 92, ay - 46), (right + 92, ay + 46)], fill=gold)
img.resize((256, 256), Image.LANCZOS).save(sys.argv[1])
