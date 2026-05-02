#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_panels_cvat_compatible.py

Base detector: detect_panels_v1.0.py (v2.2 refined core)

What this version adds:
  1) CSV output compatible with the user's CVAT-derived annotations.csv:
         image_filename,x1,y1,x2,y2
     with NO header by default.
  2) Optional batch processing over a directory or glob of images.
  3) Optional overlay drawing as before.
  4) Optional JSON output if desired for debugging.

Typical usage:
  python detect_panels_cvat_compatible.py page.jpg --csv predictions.csv
  python detect_panels_cvat_compatible.py images/ --csv predictions.csv --draw-dir overlays/

Notes:
  - Coordinates are written in the same order as the CVAT CSV: filename,x1,y1,x2,y2
  - Filename is the image basename only (e.g., FF006_03.jpg), matching your sample CSV.
  - Coordinates are written as floats with 2 decimal places to align visually with CVAT export,
    though the underlying detections are integer pixel boxes.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

Rect = Tuple[int, int, int, int]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


# --------------------------- helpers ---------------------------

def _area(r: Rect) -> int:
    x1, y1, x2, y2 = r
    return max(0, (x2 - x1)) * max(0, (y2 - y1))


def _iou(a: Rect, b: Rect) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = _area(a) + _area(b) - inter
    return inter / ua if ua > 0 else 0.0


def _contains(outer: Rect, inner: Rect, margin: int = 3) -> bool:
    x1o, y1o, x2o, y2o = outer
    x1i, y1i, x2i, y2i = inner
    return (
        (x1o - margin) <= x1i
        and (y1o - margin) <= y1i
        and (x2o + margin) >= x2i
        and (y2o + margin) >= y2i
    )


def _drop_small_contained(rects: List[Rect], ratio_thresh: float = 0.60) -> List[Rect]:
    """Remove small boxes fully contained within a larger one."""
    keep = [True] * len(rects)
    for i, ri in enumerate(rects):
        ai = _area(ri)
        for j, rj in enumerate(rects):
            if i == j or not keep[i]:
                continue
            if _contains(rj, ri) and ai < ratio_thresh * _area(rj):
                keep[i] = False
                break
    return [r for k, r in enumerate(rects) if keep[k]]


def _nms_iou(rects: List[Rect], iou_thresh: float = 0.88) -> List[Rect]:
    """Merge near-duplicate rectangles (keep largest area)."""
    rects = sorted(rects, key=_area, reverse=True)
    keep: List[Rect] = []
    for r in rects:
        if any(_iou(r, q) >= iou_thresh for q in keep):
            continue
        keep.append(r)
    return keep


def _sort_reading_order(rects: List[Rect], row_eps: int = 24) -> List[Rect]:
    if not rects:
        return rects
    pts = [(((r[1] + r[3]) // 2), ((r[0] + r[2]) // 2), r) for r in rects]
    pts.sort(key=lambda t: t[0])
    rows, cur = [], []
    for i, it in enumerate(pts):
        if i == 0:
            cur = [it]
            continue
        if abs(it[0] - cur[-1][0]) <= row_eps:
            cur.append(it)
        else:
            rows.append(cur)
            cur = [it]
    if cur:
        rows.append(cur)
    out: List[Rect] = []
    for row in rows:
        row.sort(key=lambda t: t[1])
        out.extend([t[2] for t in row])
    return out


def _remove_page_border(rects: List[Rect], W: int, H: int, min_ratio: float = 0.80) -> List[Rect]:
    if not rects:
        return rects
    page_area = W * H
    best_i, best_area = -1, 0
    for i, (x1, y1, x2, y2) in enumerate(rects):
        area = _area((x1, y1, x2, y2))
        if area > best_area and x1 <= 10 and y1 <= 10 and abs(W - 1 - x2) <= 10 and abs(H - 1 - y2) <= 10:
            best_area, best_i = area, i
    if best_i != -1 and best_area > min_ratio * page_area:
        return [r for i, r in enumerate(rects) if i != best_i]
    return rects


# --------------------------- detector ---------------------------

def detect_panels_v22_refined(image_path: str) -> List[Rect]:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    den = cv2.bilateralFilter(gray, d=7, sigmaColor=40, sigmaSpace=40)
    edges = cv2.Canny(den, 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_area = float(W * H)
    min_area = 0.03 * img_area
    rects: List[Rect] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > 0.995 * img_area:
            continue
        peri = cv2.arcLength(cnt, True)
        eps = max(2.0, 0.015 * peri)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < 4 or len(approx) > 8 or not cv2.isContourConvex(approx):
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if w >= 40 and h >= 40:
            rects.append((x, y, x + w, y + h))

    rects = _remove_page_border(rects, W, H)
    if not rects:
        return []

    rects = _drop_small_contained(rects, ratio_thresh=0.60)
    rects = _nms_iou(rects, iou_thresh=0.88)
    rects = _sort_reading_order(rects)
    return rects


# --------------------------- output helpers ---------------------------

def _draw(image_path: str, rects: List[Rect], out_path: str) -> None:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    H, W = img.shape[:2]

    def label_style() -> Tuple[float, int]:
        base = min(H, W) / 1000.0
        fs = max(0.9, min(3.5, 1.1 * base))
        th = max(2, int(round(fs * 2)))
        return fs, th

    for i, (x1, y1, x2, y2) in enumerate(rects, start=1):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), max(2, int(min(H, W) / 800)))

        fs, th = label_style()
        label = str(i)
        (tw, th_text), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        tx = max(6, min(x1 + 10, W - tw - 6))
        ty = max(6 + th_text, min(y1 + 10 + th_text, H - 6))

        pad = 6
        bg_tl = (tx - pad, ty - th_text - baseline - pad)
        bg_br = (tx + tw + pad, ty + pad)
        cv2.rectangle(img, bg_tl, bg_br, (0, 0, 0), thickness=-1)
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 0), th, cv2.LINE_AA)

    cv2.imwrite(out_path, img)


def _iter_images(input_path: str) -> List[Path]:
    p = Path(input_path)

    if p.is_file():
        return [p]

    if p.is_dir():
        return sorted([q for q in p.iterdir() if q.is_file() and q.suffix.lower() in IMAGE_EXTS])

    # Allow shell-style globs such as "images/*.jpg"
    matches = sorted(Path(m) for m in glob.glob(input_path))
    matches = [m for m in matches if m.is_file() and m.suffix.lower() in IMAGE_EXTS]
    if matches:
        return matches

    raise FileNotFoundError(f"No image(s) found for input: {input_path}")


def _csv_rows_for_image(image_name: str, rects: Sequence[Rect]) -> List[List[str]]:
    rows: List[List[str]] = []
    for x1, y1, x2, y2 in rects:
        rows.append([
            image_name,
            f"{x1:.2f}",
            f"{y1:.2f}",
            f"{x2:.2f}",
            f"{y2:.2f}",
        ])
    return rows


def _write_csv(rows: Sequence[Sequence[str]], csv_path: str, append: bool = False) -> None:
    mode = "a" if append else "w"
    with open(csv_path, mode, newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)


# --------------------------- CLI ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Detect comic panels and emit CSV compatible with the CVAT-derived annotations.csv format."
    )
    ap.add_argument(
        "input",
        help="Path to one image, a directory of images, or a glob pattern such as 'images/*.jpg'",
    )
    ap.add_argument(
        "--csv",
        metavar="OUT.csv",
        help="Write predictions in CVAT-compatible CSV format: image_filename,x1,y1,x2,y2",
    )
    ap.add_argument(
        "--append",
        action="store_true",
        help="Append to the CSV instead of overwriting it.",
    )
    ap.add_argument(
        "--json",
        metavar="OUT.json",
        help="Also write detections as JSON for debugging or downstream use.",
    )
    ap.add_argument(
        "--draw",
        metavar="OUT.png",
        help="For single-image input: write an overlay image with detected boxes.",
    )
    ap.add_argument(
        "--draw-dir",
        metavar="DIR",
        help="For batch input: write overlay images into this directory.",
    )
    args = ap.parse_args()

    image_paths = _iter_images(args.input)
    all_json = []
    all_csv_rows: List[List[str]] = []

    if len(image_paths) > 1 and args.draw:
        raise SystemExit("--draw is only for a single input image. Use --draw-dir for batch mode.")

    if args.draw_dir:
        Path(args.draw_dir).mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        rects = detect_panels_v22_refined(str(image_path))
        image_name = image_path.name

        all_csv_rows.extend(_csv_rows_for_image(image_name, rects))
        all_json.append({"image": image_name, "predictions": [list(r) for r in rects]})

        if args.draw:
            _draw(str(image_path), rects, args.draw)
        elif args.draw_dir:
            out_path = Path(args.draw_dir) / image_name
            _draw(str(image_path), rects, str(out_path))

    if args.csv:
        _write_csv(all_csv_rows, args.csv, append=args.append)
    elif len(image_paths) == 1:
        # Preserve a useful terminal default for single-image use.
        for row in all_csv_rows:
            print(",".join(row))
    else:
        raise SystemExit("Batch mode requires --csv so predictions are saved somewhere.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(all_json, fh, indent=2)


if __name__ == "__main__":
    main()
