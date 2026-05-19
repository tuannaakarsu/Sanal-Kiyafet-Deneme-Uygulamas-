"""
kiyafetigiydir.py
-----------------
Kiyafeti kisi uzerine giydiren warp modulu.
Piecewise affine warp kullanir — perspektif yerine daha esnek.
"""

import cv2
import numpy as np


# ── Yardimci: Maske Temizleme ─────────────────────────────────────────────────
def _clean_mask(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

    # En buyuk bileseni tut
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        keep = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = np.where(labels == keep, 255, 0).astype(np.uint8)

    # Delikleri doldur
    h, w = m.shape
    ff = m.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    m = cv2.bitwise_or(m, cv2.bitwise_not(ff))
    return m


# ── Yardimci: Noktadan Noktaya Affine Warp ───────────────────────────────────
def _piecewise_affine_warp(src_img, src_mask, src_pts, dst_pts, out_size):
    """
    Kaynak noktalar → hedef noktalar arasi piecewise affine warp.
    Delaunay ucgenleme ile yapilir.
    """
    h, w = out_size[1], out_size[0]
    dst_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    dst_msk    = np.zeros((h, w),    dtype=np.uint8)

    all_pts  = np.vstack([dst_pts, [[0,0],[w-1,0],[w-1,h-1],[0,h-1]]]).astype(np.float32)
    rect     = (0, 0, w, h)
    subdiv   = cv2.Subdiv2D(rect)
    for p in all_pts:
        subdiv.insert((float(p[0]), float(p[1])))

    triangles = subdiv.getTriangleList().astype(np.float32)

    # src_pts → dst_pts eslesmesi icin KDTree benzeri arama
    def find_src(dp):
        dists = np.linalg.norm(dst_pts - dp, axis=1)
        idx   = np.argmin(dists)
        return src_pts[idx] if dists[idx] < 50 else dp

    for tri in triangles:
        d = tri.reshape(3, 2)
        # Tum koseler goruntu siniri icinde olmali
        if np.any(d < 0) or np.any(d[:, 0] >= w) or np.any(d[:, 1] >= h):
            continue
        s = np.array([find_src(p) for p in d], dtype=np.float32)

        # Affine donusum: src ucgeni → dst ucgeni
        M = cv2.getAffineTransform(s, d)

        # Kaynak patch
        s_int = s.astype(np.int32)
        x, y, bw, bh = cv2.boundingRect(s_int)
        x  = max(0, x);  y  = max(0, y)
        x2 = min(src_img.shape[1], x + bw)
        y2 = min(src_img.shape[0], y + bh)
        if x2 <= x or y2 <= y:
            continue

        # Hedef ucgen maskesi
        tri_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(tri_mask, d.astype(np.int32), 255)

        # Warp
        warped = cv2.warpAffine(src_img,  M, (w, h), flags=cv2.INTER_LINEAR)
        warpedm= cv2.warpAffine(src_mask, M, (w, h), flags=cv2.INTER_NEAREST)

        warped  = cv2.bitwise_and(warped,  warped,  mask=tri_mask)
        warpedm = cv2.bitwise_and(warpedm, warpedm, mask=tri_mask)

        # Tuvale yaz (oncekinin uzerine)
        inv = cv2.bitwise_not(tri_mask)
        dst_canvas = cv2.bitwise_and(dst_canvas, dst_canvas, mask=inv)
        dst_canvas = cv2.add(dst_canvas, warped)
        dst_msk    = cv2.bitwise_or(dst_msk, warpedm)

    return dst_canvas, dst_msk


# ── Overlay ───────────────────────────────────────────────────────────────────
def overlay(person_img, warped_clothes, warped_mask):
    if warped_mask.ndim == 3:
        warped_mask = cv2.cvtColor(warped_mask, cv2.COLOR_BGR2GRAY)
    alpha = warped_mask.astype(np.float32) / 255.0
    alpha = np.expand_dims(alpha, axis=2)
    return (alpha * warped_clothes + (1 - alpha) * person_img).clip(0, 255).astype(np.uint8)


# ── Ana Warp Fonksiyonu ───────────────────────────────────────────────────────
def warp_clothes(person_img, clothes_img, mask, keypoints, widen_ratio=0.3):
    """
    Kiyafeti kisi uzerine piecewise affine warp ile giydiren ana fonksiyon.

    Parametreler
    ------------
    person_img   : BGR kisi gorseli (H x W x 3)
    clothes_img  : BGR kiyafet gorseli (H x W x 3), zaten segmente edilmis
    mask         : uint8 kiyafet maskesi (H x W)
    keypoints    : MediaPipe Pose keypoint listesi (en az 25 nokta)
    widen_ratio  : Omuz genisletme orani

    Donus
    -----
    (warped_cloth_bgr, warped_mask_uint8) veya (None, None)
    """
    # --- Keypoint al ---
    try:
        ls = np.array(keypoints[11][:2], dtype=np.float32)   # sol omuz
        rs = np.array(keypoints[12][:2], dtype=np.float32)   # sag omuz
        le = np.array(keypoints[13][:2], dtype=np.float32)   # sol dirsek
        re = np.array(keypoints[14][:2], dtype=np.float32)   # sag dirsek
        lw = np.array(keypoints[15][:2], dtype=np.float32)   # sol bilek
        rw = np.array(keypoints[16][:2], dtype=np.float32)   # sag bilek
        lh = np.array(keypoints[23][:2], dtype=np.float32)   # sol kalca
        rh = np.array(keypoints[24][:2], dtype=np.float32)   # sag kalca
    except (IndexError, TypeError):
        return None, None

    src_mask = _clean_mask(mask)
    ys, xs   = np.where(src_mask > 0)
    if len(xs) == 0:
        return None, None

    x, y, bw, bh = cv2.boundingRect(np.stack([xs, ys], axis=1))
    if bw < 4 or bh < 4:
        return None, None

    hp, wp = person_img.shape[:2]
    out_size = (wp, hp)

    # --- Kaynak noktalar (cloth product image space) ---
    x0, y0 = float(x), float(y)
    x1, y1 = float(x + bw), float(y + bh)
    cx      = (x0 + x1) * 0.5

    src_pts = np.array([
        [cx,              y0],            # ust merkez (yaka)
        [x0 + bw*0.22,   y0 + bh*0.05],  # sol yaka
        [x1 - bw*0.22,   y0 + bh*0.05],  # sag yaka
        [x0 + bw*0.05,   y0 + bh*0.15],  # sol omuz ustu
        [x1 - bw*0.05,   y0 + bh*0.15],  # sag omuz ustu
        [x0,             y0 + bh*0.25],  # sol omuz dis
        [x1,             y0 + bh*0.25],  # sag omuz dis
        [x0,             y0 + bh*0.55],  # sol kol orta
        [x1,             y0 + bh*0.55],  # sag kol orta
        [x0 + bw*0.08,   y0 + bh*0.85],  # sol bel
        [x1 - bw*0.08,   y0 + bh*0.85],  # sag bel
        [x0 + bw*0.15,   y1],            # sol alt
        [x1 - bw*0.15,   y1],            # sag alt
        [cx,             y1],            # alt merkez
    ], dtype=np.float32)

    # --- Hedef noktalar (vucuttaki karsilik gelen konumlar) ---
    shoulder_vec = rs - ls
    hip_vec      = rh - lh
    arm_w        = max(10.0, np.linalg.norm(shoulder_vec) * 0.13)

    lsw = ls - shoulder_vec * widen_ratio        # genisletilmis sol omuz
    rsw = rs + shoulder_vec * widen_ratio        # genisletilmis sag omuz
    lhw = lh - hip_vec * (widen_ratio * 0.5)    # genisletilmis sol kalca
    rhw = rh + hip_vec * (widen_ratio * 0.5)    # genisletilmis sag kalca

    mid_top    = (lsw + rsw) * 0.5
    mid_bottom = (lhw + rhw) * 0.5

    # Kol noktalarinin normali
    def _arm_side(a, b, c, side, w):
        """side: +1 sol, -1 sag"""
        v   = b - a
        n   = np.array([-v[1], v[0]], dtype=np.float32)
        n  /= np.linalg.norm(n) + 1e-6
        return a + n * w * side, c + n * w * side

    ls_out, le_out = _arm_side(lsw, le, le, +1, arm_w)
    rs_out, re_out = _arm_side(rsw, re, re, -1, arm_w)

    dst_pts = np.array([
        mid_top,                          # ust merkez
        lsw + (rsw - lsw) * 0.22,        # sol yaka
        rsw - (rsw - lsw) * 0.22,        # sag yaka
        lsw + (rsw - lsw) * 0.05,        # sol omuz ustu
        rsw - (rsw - lsw) * 0.05,        # sag omuz ustu
        lsw,                              # sol omuz dis
        rsw,                             # sag omuz dis
        ls_out,                          # sol kol orta
        rs_out,                          # sag kol orta
        lhw + (rhw - lhw) * 0.08,        # sol bel
        rhw - (rhw - lhw) * 0.08,        # sag bel
        lhw + (rhw - lhw) * 0.15,        # sol alt
        rhw - (rhw - lhw) * 0.15,        # sag alt
        mid_bottom,                      # alt merkez
    ], dtype=np.float32)

    # --- Piecewise affine warp ---
    warped_img, warped_msk = _piecewise_affine_warp(
        clothes_img, src_mask, src_pts, dst_pts, out_size
    )

    # --- Torso ROI disina tasmayi engelle ---
    torso_poly = np.array([lsw, rsw, rhw, lhw], dtype=np.int32)
    torso_roi  = np.zeros((hp, wp), dtype=np.uint8)
    cv2.fillConvexPoly(torso_roi, torso_poly, 255)
    torso_roi  = cv2.dilate(torso_roi, np.ones((30, 30), np.uint8), iterations=1)
    warped_msk = cv2.bitwise_and(warped_msk.astype(np.uint8), torso_roi)
    warped_img = cv2.bitwise_and(warped_img, warped_img, mask=warped_msk)

    # --- Kenar yumusat ---
    warped_msk = _clean_mask(warped_msk)
    warped_msk = cv2.GaussianBlur(warped_msk, (0, 0), sigmaX=2.0)
    warped_msk = (warped_msk > 20).astype(np.uint8) * 255
    warped_img = cv2.bitwise_and(warped_img, warped_img, mask=warped_msk)

    return warped_img, warped_msk