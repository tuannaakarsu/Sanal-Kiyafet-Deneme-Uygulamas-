import cv2
import numpy as np
import torch
from posetespiti import PoseEstimator
from segmentasyon import load_model, segment_cloth
from kıyafetigiydir import warp_clothes, overlay

def apply_mask(image, mask):
    # Mask tipini ve boyutunu kontrol et
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    return cv2.bitwise_and(image, image, mask=mask)

def run_virtual_tryon_auto(person_img_path, clothes_img_path, unet_model_path,output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Görselleri oku
    person_img = cv2.imread(person_img_path)
    clothes_img = cv2.imread(clothes_img_path)

    if person_img is None or clothes_img is None:
        raise ValueError("Görseller yüklenemedi. Dosya yolları doğru mu?")

    # Segmentasyon modeli yükle
    unet_model = load_model(unet_model_path, device)
    binary_mask = segment_cloth(clothes_img, unet_model, device)
    segmented_clothes = apply_mask(clothes_img, binary_mask)
    # ✅ Boyut eşitle

    segmented_clothes = cv2.resize(segmented_clothes, (person_img.shape[1], person_img.shape[0]))
    binary_mask = cv2.resize(binary_mask, (person_img.shape[1], person_img.shape[0]))

# Poz tahmini
    pose_estimator = PoseEstimator()
    keypoints, _ = pose_estimator.get_keypoints(person_img)

    # Giydirme işlemi
    if keypoints is not None:
        try:
            warped_clothes, warped_mask = warp_clothes(
                person_img,
                segmented_clothes,
                binary_mask,
                keypoints,
                widen_ratio=0.3
            )
            result = overlay(person_img, warped_clothes, warped_mask)

        except Exception as e:
            print("⚠️ Warp sırasında hata:", e)
            result = blended_fallback(person_img, segmented_clothes, binary_mask)
    else:
        print("⚠️ Anahtar noktalar bulunamadı. Fallback blending uygulanıyor.")
        result = blended_fallback(person_img, segmented_clothes, binary_mask)
    cv2.imwrite(output_path, result)
    return result

def blended_fallback(person_img, clothing_img, mask):

    alpha = mask.astype(float) / 255.0
    alpha = np.stack([alpha] * 3, axis=-1)
    blended = (alpha * clothing_img + (1 - alpha) * person_img).astype(np.uint8)
    return blended
