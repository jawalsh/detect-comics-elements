# Comics Element Detector

This guide introduces a Python program that detects rectangular **comic panels** in a scanned page image and outputs each panel’s bounding box as `[x1, y1, x2, y2]`. It also draws an optional debug overlay so you can verify detections visually. The document includes **installation instructions**, **how to run**, the **entire program code**, and a **chunk‑by‑chunk walkthrough** explaining what every important part does in plain language.

---

## 🎯 What the program does

Input: a page image (e.g., `page.jpg`).  
Output: a JSON array like:

```json
[
  [x1, y1, x2, y2],
  [x1, y1, x2, y2],
  ...
]
```

Each list gives the upper‑left `(x1, y1)` and lower‑right `(x2, y2)` corners of a detected panel. Optionally, a **debug image** is written with green rectangles numbered in reading order.

This “v2.2‑refined” variant is tuned for pages with **closed, high‑contrast borders**. It adds two post‑processing steps that improve accuracy:

1. **Containment filter** – removes small rectangles inside larger ones (like captions).  
2. **Duplicate merge (NMS)** – merges near‑duplicate rectangles that overlap heavily.

---

## 🧰 Installation

Install Python and dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install opencv-python-headless numpy
```

Or on Ubuntu/Debian:

```bash
sudo apt install python3-opencv python3-numpy
```

---

## 🚀 How to run

```bash
python detect_panels_v22_refined.py page.jpg --draw debug.png
```

This prints panel coordinates to the terminal and optionally writes a `debug.png` overlay with green rectangles.

![Sample png output with panels highlighted and numbered.](samples/output.png)

---

## 📚 Libraries used

| Library | Purpose | Docs |
|----------|----------|------|
| OpenCV (`cv2`) | Image processing and contour detection | https://opencv.org |
| NumPy (`np`) | Array math for coordinates and images | https://numpy.org |
| argparse | Handles command-line arguments | https://docs.python.org/3/library/argparse.html |
| json | Prints structured JSON output | https://docs.python.org/3/library/json.html |

---

## 🧠 Code Walkthrough (with examples)

### 1️⃣ Imports and type alias

```python
import argparse, json
from typing import List, Tuple
import cv2, numpy as np
Rect = Tuple[int,int,int,int]
```

- `argparse` lets the user specify an image and optional `--draw` output file.  
- `json` outputs panel coordinates as JSON.  
- `cv2` (OpenCV) handles image processing.  
- `numpy` provides fast math on image data.  
- `Rect` defines a rectangle as a 4-tuple `(x1, y1, x2, y2)`.

---

### 2️⃣ Geometry helpers

```python
def _area(r: Rect) -> int:
    x1, y1, x2, y2 = r
    return max(0, (x2-x1)) * max(0, (y2-y1))
```

Computes the area of a rectangle in pixels.

```python
def _iou(a: Rect, b: Rect) -> float:
    # Intersection-over-Union (IoU)
```
Measures how much two rectangles overlap. Used to merge duplicates.

```python
def _contains(outer, inner, margin=3) -> bool:
    # Checks if inner lies fully inside outer
```

Used to remove caption boxes inside panels.

---

### 3️⃣ Containment and duplicate cleanup

```python
_drop_small_contained(rects, ratio_thresh=0.60)
```
Removes rectangles smaller than 60% of their containing rectangle’s area.

```python
_nms_iou(rects, iou_thresh=0.88)
```
Merges highly overlapping rectangles to eliminate duplicates.

---

### 4️⃣ Sorting by reading order

```python
_sort_reading_order(rects, row_eps=24)
```
Groups panels into rows (within 24 pixels vertically) and sorts each row left to right.

---

### 5️⃣ Page border removal

```python
_remove_page_border(rects, W, H, min_ratio=0.8)
```
Removes any detected rectangle that matches nearly the full page (≥80% of image area).

---

### 6️⃣ Main detection pipeline

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
den = cv2.bilateralFilter(gray, d=7, sigmaColor=40, sigmaSpace=40)
edges = cv2.Canny(den, 60, 180)
edges = cv2.dilate(edges, np.ones((3,3), np.uint8), 1)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)), 1)
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
```

This sequence:
- Smooths noise while keeping edges.  
- Finds edges with Canny.  
- Thickens them (dilate).  
- Closes small gaps (morphology).  
- Extracts contour outlines for panel candidates.

Then filters by area and convexity to keep valid rectangles.

---

### 7️⃣ Drawing debug overlays

```python
cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 3)
cv2.putText(img, str(i), (x1+6,y1+28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 1)
```

Draws numbered green rectangles on a copy of the input page.

---

### 8️⃣ Command-line interface (CLI)

```python
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--draw", metavar="OUT.png")
    args = ap.parse_args()
    rects = detect_panels_v22_refined(args.image)
    print(json.dumps([list(r) for r in rects], indent=2))
    if args.draw:
        _draw(args.image, rects, args.draw)
```

Lets you run the program from a terminal, outputs JSON, and optionally writes the overlay.

---

## 🧾 Example Output

```json
[
  [35, 120, 2980, 560],
  [40, 620, 1460, 1200],
  [1520, 620, 2980, 1200],
  [40, 1260, 1460, 1920],
  [1520, 1260, 2980, 1920]
]
```

To crop one panel using ImageMagick:

```bash
convert page.jpg -crop "2945x440+35+120" +repage panel_1.jpg
```

---

## 🧩 Troubleshooting

| Problem | Cause | Fix |
|----------|--------|-----|
| `ModuleNotFoundError: cv2` | Missing OpenCV | `pip install opencv-python-headless` |
| Empty output | Weak borders | Lower Canny thresholds (50,150) |
| Too many boxes | Increase `min_area` or ratio threshold |  |
| Duplicate panels | Lower `iou_thresh` to 0.85 |  |

---

## 🔮 Next Steps

- Extend detection to **non-rectangular panels** (polygons, ellipses).  
- Add **color-based** detection for comics without black borders.  
- Train a **machine-learning model** (Mask R-CNN or YOLO) using this script’s results.  
- Batch process entire collections with a single command.

---
