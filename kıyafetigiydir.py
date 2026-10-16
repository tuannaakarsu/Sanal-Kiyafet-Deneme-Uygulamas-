"""
kiyafetigiydir.py - v3
"""
print("kiyafetigiydir.py v3 YUKLENDI")

import cv2
import numpy as np


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        keep = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = np.where(labels == keep, 255, 0).astype(np.uint8)
    h, w = m.shape
    ff = m.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    m = cv2.bitwise_or(m, cv2.bitwise_not(ff))
    return m


def overlay(person_img, warped_clothes, warped_mask):
    if warped_mask.ndim == 3:
        warped_mask = cv2.cvtColor(warped_mask, cv2.COLOR_BGR2GRAY)
    alpha = warped_mask.astype(np.float32) / 255.0
    alpha = np.expand_dims(alpha, axis=2)
    return (alpha * warped_clothes + (1 - alpha) * person_img).clip(0, 255).astype(np.uint8)


def warp_clothes(person_img, clothes_img, mask, keypoints, widen_ratio=0.3):
    print("warp_clothes CAGIRILDI")

    try:
        ls = np.array(keypoints[11][:2], dtype=np.float32)
        rs = np.array(keypoints[12][:2], dtype=np.float32)
        lh = np.array(keypoints[23][:2], dtype=np.float32)
        rh = np.array(keypoints[24][:2], dtype=np.float32)
    except (IndexError, TypeError) as e:
        print(f"WARP: keypoint hatasi: {e}")
        return None, None

    print(f"WARP: ls={ls} rs={rs} lh={lh} rh={rh}")

    hp, wp = person_img.shape[:2]

    all_kps   = np.array([ls, rs, lh, rh])
    kp_y_min  = all_kps[:, 1].min()
    kp_y_max  = all_kps[:, 1].max()
    kp_x_min  = all_kps[:, 0].min()
    kp_x_max  = all_kps[:, 0].max()
    kp_height = kp_y_max - kp_y_min
    kp_width  = kp_x_max - kp_x_min

    print(f"WARP: kp_height={kp_height:.1f} kp_width={kp_width:.1f} img={hp}x{wp}")

    if kp_height < hp * 0.4:
        scale = min((hp * 0.55) / max(kp_height, 1), 1.8)
        cx    = wp / 2
        cy    = hp * 0.38
        kp_cx = kp_x_min + kp_width / 2
        kp_cy = kp_y_min + kp_height / 2

        def rescale(pt):
            return np.array([
                cx + (pt[0] - kp_cx) * scale,
                cy + (pt[1] - kp_cy) * scale,
            ], dtype=np.float32)

        ls = rescale(ls)
        rs = rescale(rs)
        lh = rescale(lh)
        rh = rescale(rh)
        print(f"WARP: scaled ls={ls} rs={rs} lh={lh} rh={rh}")

    if ls[0] < rs[0]:
        ls, rs = rs.copy(), ls.copy()
    if lh[0] < rh[0]:
        lh, rh = rh.copy(), lh.copy()

    shoulder_vec   = rs - ls
    hip_vec        = rh - lh
    shoulder_width = np.linalg.norm(shoulder_vec)
    min_width      = wp * 0.42

    if shoulder_width < min_width:
        sf   = min_width / max(shoulder_width, 1)
        cx   = (ls[0] + rs[0]) / 2
        cy_s = (ls[1] + rs[1]) / 2
        cy_h = (lh[1] + rh[1]) / 2
        ls   = np.array([cx + (ls[0] - cx) * sf, cy_s], dtype=np.float32)
        rs   = np.array([cx + (rs[0] - cx) * sf, cy_s], dtype=np.float32)
        lh   = np.array([cx + (lh[0] - cx) * sf, cy_h], dtype=np.float32)
        rh   = np.array([cx + (rh[0] - cx) * sf, cy_h], dtype=np.float32)
        shoulder_vec = rs - ls
        hip_vec      = rh - lh

    src_mask = _clean_mask(mask)
    ys, xs   = np.where(src_mask > 0)
    if len(xs) == 0:
        print("WARP: maske bos")
        return None, None

    x, y, bw, bh = cv2.boundingRect(np.stack([xs, ys], axis=1))
    print(f"WARP: cloth bbox x={x} y={y} w={bw} h={bh}")

    src_pts = np.array([
    [x,      y     ],   # sol üst
    [x + bw, y     ],   # sağ üst
    [x + bw, y + bh],   # sağ alt
    [x,      y + bh],   # sol alt
    ], dtype=np.float32)

    lsw = ls + np.array([shoulder_vec[1] * 0, -0.0]) + np.array([widen_ratio * np.linalg.norm(shoulder_vec), 0])
    rsw = rs - np.array([widen_ratio * np.linalg.norm(shoulder_vec), 0])
    lhw = lh + np.array([widen_ratio * 0.4 * np.linalg.norm(hip_vec), 0])
    rhw = rh - np.array([widen_ratio * 0.4 * np.linalg.norm(hip_vec), 0])

    dst_pts = np.array([
    lsw,   # sol üst (büyük x) → src sol üst
    rsw,   # sağ üst (küçük x) → src sağ üst
    rhw,   # sağ alt → src sağ alt
    lhw,   # sol alt → src sol alt
    ], dtype=np.float32)

    print(f"WARP: dst_pts rsw={rsw} lsw={lsw} lhw={lhw} rhw={rhw}") 

    M           = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_img  = cv2.warpPerspective(clothes_img, M, (wp, hp), flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpPerspective(src_mask,    M, (wp, hp), flags=cv2.INTER_NEAREST)
    warped_mask = (warped_mask > 0).astype(np.uint8) * 255

    torso_poly  = np.array([lsw, rsw, rhw, lhw], dtype=np.int32)
    torso_roi   = np.zeros((hp, wp), dtype=np.uint8)
    cv2.fillConvexPoly(torso_roi, torso_poly, 255)
    torso_roi   = cv2.dilate(torso_roi, np.ones((25, 25), np.uint8), iterations=1)
    warped_mask = cv2.bitwise_and(warped_mask, torso_roi)
    warped_img  = cv2.bitwise_and(warped_img, warped_img, mask=warped_mask)

    warped_mask = cv2.GaussianBlur(warped_mask, (0, 0), sigmaX=2.0)
    warped_mask = (warped_mask > 20).astype(np.uint8) * 255
    warped_img  = cv2.bitwise_and(warped_img, warped_img, mask=warped_mask)

    print(f"WARP: tamamlandi, warped_mask sum={warped_mask.sum()}")
    return warped_img, warped_mask