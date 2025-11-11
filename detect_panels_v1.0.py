#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_panels_v1.0.py
Base: v2.2 (columns + valley-split)
Refinements:
  1) _drop_small_contained(): removes inner text boxes (area ratio < 0.6).
  2) _nms_iou(): merges near-duplicate overlapping boxes (IoU >= 0.88).
"""

from __future__ import annotations
import argparse, json
from typing import List, Tuple
import cv2, numpy as np

Rect = Tuple[int,int,int,int]

# --------------------------- helpers ---------------------------

def _area(r: Rect) -> int:
    x1,y1,x2,y2 = r
    return max(0,(x2-x1)) * max(0,(y2-y1))

def _iou(a: Rect, b: Rect) -> float:
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1)
    ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1)
    inter=iw*ih
    ua=_area(a)+_area(b)-inter
    return inter/ua if ua>0 else 0.0

def _contains(outer: Rect, inner: Rect, margin=3) -> bool:
    x1o,y1o,x2o,y2o=outer
    x1i,y1i,x2i,y2i=inner
    return (x1o - margin)<=x1i and (y1o - margin)<=y1i and (x2o + margin)>=x2i and (y2o + margin)>=y2i

def _drop_small_contained(rects: List[Rect], ratio_thresh=0.60) -> List[Rect]:
    """Remove small boxes fully contained within a larger one."""
    keep=[True]*len(rects)
    for i,ri in enumerate(rects):
        ai=_area(ri)
        for j,rj in enumerate(rects):
            if i==j or not keep[i]: continue
            if _contains(rj,ri) and ai < ratio_thresh*_area(rj):
                keep[i]=False
                break
    return [r for k,r in enumerate(rects) if keep[k]]

def _nms_iou(rects: List[Rect], iou_thresh=0.88) -> List[Rect]:
    """Merge near-duplicate rectangles (keep largest area)."""
    rects=sorted(rects,key=_area,reverse=True)
    keep=[]
    for r in rects:
        if any(_iou(r,q)>=iou_thresh for q in keep):
            continue
        keep.append(r)
    return keep

def _sort_reading_order(rects: List[Rect], row_eps: int = 24) -> List[Rect]:
    if not rects: return rects
    pts=[(((r[1]+r[3])//2),((r[0]+r[2])//2),r) for r in rects]
    pts.sort(key=lambda t:t[0])
    rows,cur=[],[]
    for i,it in enumerate(pts):
        if i==0: cur=[it]; continue
        if abs(it[0]-cur[-1][0])<=row_eps: cur.append(it)
        else: rows.append(cur); cur=[it]
    if cur: rows.append(cur)
    out=[]
    for row in rows:
        row.sort(key=lambda t:t[1])
        out.extend([t[2] for t in row])
    return out

def _remove_page_border(rects: List[Rect], W:int, H:int, min_ratio=0.80) -> List[Rect]:
    if not rects: return rects
    page_area=W*H
    best_i,best_area=-1,0
    for i,(x1,y1,x2,y2) in enumerate(rects):
        area=_area((x1,y1,x2,y2))
        if area>best_area and x1<=10 and y1<=10 and abs(W-1-x2)<=10 and abs(H-1-y2)<=10:
            best_area,best_i=area,i
    if best_i!=-1 and best_area>min_ratio*page_area:
        return [r for i,r in enumerate(rects) if i!=best_i]
    return rects

# --------------------------- main detector (v2.2 core) ---------------------------

def detect_panels_v22_refined(image_path: str) -> List[Rect]:
    img=cv2.imread(image_path,cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(image_path)
    H,W=img.shape[:2]
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    den=cv2.bilateralFilter(gray,d=7,sigmaColor=40,sigmaSpace=40)
    edges=cv2.Canny(den,60,180)
    edges=cv2.dilate(edges,np.ones((3,3),np.uint8),iterations=1)
    edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),iterations=1)
    contours,_=cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    img_area=float(W*H)
    min_area=0.03*img_area
    rects=[]
    for cnt in contours:
        area=float(cv2.contourArea(cnt))
        if area<min_area or area>0.995*img_area:
            continue
        peri=cv2.arcLength(cnt,True)
        eps=max(2.0,0.015*peri)
        approx=cv2.approxPolyDP(cnt,eps,True)
        if len(approx)<4 or len(approx)>8 or not cv2.isContourConvex(approx):
            continue
        x,y,w,h=cv2.boundingRect(approx)
        if w>=40 and h>=40:
            rects.append((x,y,x+w,y+h))

    rects=_remove_page_border(rects,W,H)
    if not rects: return []

    # Sort & cleanup (only light containment + NMS)
    rects=_drop_small_contained(rects,ratio_thresh=0.60)
    rects=_nms_iou(rects,iou_thresh=0.88)
    rects=_sort_reading_order(rects)
    return rects

# --------------------------- draw & CLI ---------------------------

def _draw(image_path: str, rects: List[Rect], out_path: str):
    img=cv2.imread(image_path,cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(image_path)
    for i,(x1,y1,x2,y2) in enumerate(rects,start=1):
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),3)
        cv2.putText(img,str(i),(x1+6,y1+28),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(img,str(i),(x1+6,y1+28),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),1,cv2.LINE_AA)
    cv2.imwrite(out_path,img)

def main():
    ap=argparse.ArgumentParser(description="Detect comic panels (v2.2 refined).")
    ap.add_argument("image")
    ap.add_argument("--draw",metavar="OUT.png")
    args=ap.parse_args()
    rects=detect_panels_v22_refined(args.image)
    print(json.dumps([list(r) for r in rects],indent=2))
    if args.draw:
        _draw(args.image,rects,args.draw)

if __name__=="__main__":
    main()