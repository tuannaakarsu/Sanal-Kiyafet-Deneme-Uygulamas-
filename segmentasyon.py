import torch
import cv2
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as T
from model import UNet


def load_model(weight_path, device):
    model = UNet(3, 59)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def segment_cloth(image, model, device):
    """ kıyafetleri segmentliyoruz."""

    # --- 1. Görüntü boyutunu kaydet ---
    orig_h, orig_w = image.shape[:2]

    # --- 2. Görüntüyü 256x256'e yeniden boyutlandır, normalize et ---
    image_resized = cv2.resize(image, (256, 256))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0  # [3, 256, 256]
    input_tensor = image_tensor.unsqueeze(0).to(device)  # [1, 3, 256, 256]

    # --- 3. Model tahmini ---
    with torch.no_grad():
        output = model(input_tensor)  # [1, 59, 256, 256]
        pred = torch.argmax(output[0], dim=0).cpu().numpy()  # [256, 256]

    # --- 4. Arka plan dışındaki en yoğun sınıfı bul ---
    unique, counts = np.unique(pred, return_counts=True)
    class_freq = dict(zip(unique, counts))
    class_freq.pop(0, None)  # Arka plan (0) çıkarılır

    if not class_freq:
        print("Uyarı: Arka plan dışında sınıf bulunamadı.")
        return np.zeros((orig_h, orig_w), dtype=np.uint8)

    dominant_class = max(class_freq, key=class_freq.get)
    print(f"Seçilen dominant sınıf ID: {dominant_class}")

    # --- 5. Binary maske oluştur ---
    binary_mask = np.where(pred == dominant_class, 255, 0).astype(np.uint8)

    # --- 6. Maske iyileştirme (dilate + en büyük bileşen) ---
    kernel = np.ones((5, 5), np.uint8)
    mask_dilated = cv2.dilate(binary_mask, kernel, iterations=5)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_dilated, connectivity=8)

    if num_labels > 1:
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_clean = np.where(labels == largest_component, 255, 0).astype(np.uint8)
    else:
        mask_clean = binary_mask

    return mask_clean

