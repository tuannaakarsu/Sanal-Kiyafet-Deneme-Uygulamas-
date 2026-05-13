import cv2
import numpy as np

def overlay(person_img, warped_clothes, warped_mask):
    if warped_mask.ndim == 3:
        warped_mask = cv2.cvtColor(warped_mask, cv2.COLOR_BGR2GRAY)

    alpha = warped_mask.astype(np.float32) / 255.0
    alpha = np.expand_dims(alpha, axis=2)

    result = (alpha * warped_clothes + (1 - alpha) * person_img).astype(np.uint8)
    return result


def warp_clothes(person_img, clothes_img, mask, keypoints, widen_ratio=0.7):
    try:
        LS = np.array(keypoints[11][:2], dtype=np.float32)
        RS = np.array(keypoints[12][:2], dtype=np.float32)
        LH = np.array(keypoints[23][:2], dtype=np.float32)
        RH = np.array(keypoints[24][:2], dtype=np.float32)
    except IndexError:
        print("⚠️ Keypoint indeksleri hatalı.")
        return person_img, None, None

    shoulder_vec = RS - LS
    hip_vec = RH - LH

    shoulder_extend = shoulder_vec * widen_ratio
    hip_extend = hip_vec * widen_ratio

    LS_wide = LS - shoulder_extend
    RS_wide = RS + shoulder_extend
    LH_wide = LH - hip_extend
    RH_wide = RH + hip_extend

    dst_pts = np.array([LS_wide, RS_wide, RH_wide, LH_wide], dtype=np.float32)

    # ✅ HATA DÜZELTİLDİ
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        print("⚠️ Maske boş olabilir.")
        return person_img, None, None

    x, y, w, h = cv2.boundingRect(np.stack([xs, ys], axis=1))
    src_pts = np.array([
        [x,     y    ],
        [x + w, y    ],
        [x + w, y + h],
        [x,     y + h]
    ], dtype=np.float32)

    src_aspect = w / h
    dst_width = np.linalg.norm(RS_wide - LS_wide)
    dst_height = np.linalg.norm(RH_wide - RS_wide)
    dst_aspect = dst_width / dst_height if dst_height > 0 else 1

    if abs(src_aspect - dst_aspect) > 0.5:
        print(f"⚠️ Kaynak ve hedef oranlar uyuşmuyor: {src_aspect:.2f} vs {dst_aspect:.2f}")

    try:
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    except cv2.error as e:
        print(f"⚠️ getPerspectiveTransform hatası: {e}")
        return person_img, None, None

    Hp, Wp = person_img.shape[:2]
    warped_clothes = cv2.warpPerspective(clothes_img, M, (Wp, Hp), flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpPerspective(mask, M, (Wp, Hp), flags=cv2.INTER_NEAREST)
    warped_mask = (warped_mask > 0).astype(np.uint8) * 255

    # 👇 Yeni ve daha doğal overlay fonksiyonu kullanılsın
    result = overlay(person_img, warped_clothes, warped_mask)

    return warped_clothes, warped_mask

