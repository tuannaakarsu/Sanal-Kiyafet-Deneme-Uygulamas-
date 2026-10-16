import cv2
import numpy as np
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image

from model import TryOnGenerator
from modelunet import UNet
from posetespiti import PoseEstimator
from kıyafetigiydir import warp_clothes


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENERATOR_PATH = "models/generator_3.pth"
UNET_PATH = "models/unet_model_2.pth"
SIZE = (256, 256)


@st.cache_resource
def load_generator():
    model = TryOnGenerator()
    model.load_state_dict(torch.load(GENERATOR_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


@st.cache_resource
def load_pose_estimator():
    return PoseEstimator(static_image_mode=True)


@st.cache_resource
def load_unet():
    model = UNet(3, 59)
    try:
        state = torch.load(UNET_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        return model
    except Exception:
        return None


norm_transform = T.Compose(
    [
        T.Resize(SIZE),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = (tensor * 0.5) + 0.5
    tensor = tensor.clamp(0, 1)
    img = tensor.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


def pil_to_bgr(img):
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def make_pose_map(person_bgr, pose_estimator):
    person_bgr = cv2.resize(person_bgr, SIZE)
    return pose_estimator.render_openpose_map(person_bgr, thickness=6, joint_radius=5)


def make_agnostic(person_bgr, pose_estimator, cloth_region_mask=None):
    person_bgr = cv2.resize(person_bgr, SIZE)
    agnostic = person_bgr.copy()
    h, w, _ = agnostic.shape

    keypoints_and_landmarks = pose_estimator.get_keypoints(person_bgr)
    if keypoints_and_landmarks is None:
        # Pose yoksa merkez govdeyi bastirarak kaba bir agnostic uret.
        x1, x2 = int(w * 0.34), int(w * 0.66)
        y1, y2 = int(h * 0.22), int(h * 0.68)
        cv2.rectangle(agnostic, (x1, y1), (x2, y2), (127, 127, 127), -1)
        return agnostic

    keypoints, _ = keypoints_and_landmarks
    # mediapipe indices: 11 LS, 12 RS, 23 LH, 24 RH
    req = [11, 12, 23, 24]
    if any(len(keypoints) <= i for i in req):
        return agnostic

    ls = keypoints[11][:2].astype(np.int32)
    rs = keypoints[12][:2].astype(np.int32)
    lh = keypoints[23][:2].astype(np.int32)
    rh = keypoints[24][:2].astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)

    # Govde poligonu (omuz + kalca), bir miktar genisletme ile.
    torso = np.array([ls, rs, rh, lh], dtype=np.int32)
    cv2.fillConvexPoly(mask, torso, 255)
    mask = cv2.dilate(mask, np.ones((11, 11), np.uint8), iterations=1)

    # Kollari da bastirmak icin omuz-dirsek-bilek cizgileri.
    arm_pairs = [(11, 13), (13, 15), (12, 14), (14, 16)]
    for a, b in arm_pairs:
        if len(keypoints) > b:
            p1 = tuple(keypoints[a][:2].astype(np.int32))
            p2 = tuple(keypoints[b][:2].astype(np.int32))
            cv2.line(mask, p1, p2, 255, 14)
            cv2.circle(mask, p1, 9, 255, -1)
            cv2.circle(mask, p2, 9, 255, -1)

    # Warp'tan gelen kiyafet bolgesini de agnostic maskesine ekle.
    if cloth_region_mask is not None:
        crm = cloth_region_mask.astype(np.uint8)
        if crm.shape != mask.shape:
            crm = cv2.resize(crm, (w, h), interpolation=cv2.INTER_NEAREST)
        crm = cv2.morphologyEx(crm, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
        mask = cv2.bitwise_or(mask, crm)

    # Bas/yuz bolgesini koru (agnosticte silinmesin).
    if len(keypoints) > 0:
        nx, ny, nv = keypoints[0]
        if nv > 0.2:
            head_keep = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(head_keep, (int(nx), int(ny)), 28, 255, -1)
            cv2.circle(head_keep, (int(nx), int(ny + 26)), 24, 255, -1)
            mask[head_keep > 0] = 0

    # Maske bolgesini notr renkle bastir.
    agnostic[mask > 0] = (127, 127, 127)
    return agnostic


def make_cloth_mask(cloth_bgr, unet_model):
    cloth_bgr = cv2.resize(cloth_bgr, SIZE)

    # Önce UNet dene
    if unet_model is not None:
        cloth_rgb = cv2.cvtColor(cloth_bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(cloth_rgb).permute(2, 0, 1).float() / 255.0
        x = x.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = unet_model(x)[0]
            pred = torch.argmax(logits, dim=0).cpu().numpy().astype(np.uint8)
        unique, counts = np.unique(pred, return_counts=True)
        freq = {int(k): int(v) for k, v in zip(unique, counts) if int(k) != 0}
        if freq:
            dominant = max(freq, key=freq.get)
            mask = np.where(pred == dominant, 255, 0).astype(np.uint8)
            h, w = mask.shape
            center = mask[h//4:3*h//4, w//4:3*w//4]
            if np.sum(center) < np.sum(mask) * 0.25:
                mask = cv2.bitwise_not(mask)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            # Maske makul görünüyorsa döndür
            coverage = np.sum(mask > 0) / (h * w)
            if 0.1 < coverage < 0.9:
                return mask

    # UNet başarısız olduysa arka plan rengine göre ayır
    # Köşe piksellerinin ortalaması arka plan rengi
    h, w = cloth_bgr.shape[:2]
    corners = np.array([
        cloth_bgr[0, 0], cloth_bgr[0, w-1],
        cloth_bgr[h-1, 0], cloth_bgr[h-1, w-1]
    ], dtype=np.float32)
    bg_color = corners.mean(axis=0)

    # Arka plana benzeyen pikselleri maskele
    diff = np.abs(cloth_bgr.astype(np.float32) - bg_color).sum(axis=2)
    mask = (diff > 30).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def run_inference(person_img, cloth_img, generator, unet_model, pose_estimator):
    person_bgr = pil_to_bgr(person_img)
    cloth_bgr  = pil_to_bgr(cloth_img)
    person_bgr = cv2.resize(person_bgr, SIZE)
    cloth_bgr  = cv2.resize(cloth_bgr,  SIZE)

    person_bgr = cv2.resize(person_bgr, SIZE)

    pose_bgr   = make_pose_map(person_bgr, pose_estimator)
    cloth_mask = make_cloth_mask(cloth_bgr, unet_model)

    segmented_cloth       = cv2.bitwise_and(cloth_bgr, cloth_bgr, mask=cloth_mask.astype(np.uint8))
    keypoints_and_landmarks = pose_estimator.get_keypoints(person_bgr)
    st.write(f"Pose tespit: {keypoints_and_landmarks is not None}")
    warped_cloth, warped_mask = segmented_cloth, cloth_mask

    if keypoints_and_landmarks is not None:
        keypoints, _ = keypoints_and_landmarks  # BU SATIR EKSİK
        st.write(f"Keypoint sayısı: {len(keypoints)}")
        try:
            wc, wm = warp_clothes(
                person_bgr,
                segmented_cloth,
                cloth_mask.astype(np.uint8),
                keypoints,
                widen_ratio=0.3,
            )
            if wc is not None and wm is not None:
                warped_cloth, warped_mask = wc, wm
        except Exception as e:
            st.error(f"Warp hatası: {e}")
            import traceback
            st.code(traceback.format_exc())

        try:
            ls = keypoints[11][:2].astype(np.int32)
            rs = keypoints[12][:2].astype(np.int32)
            lh = keypoints[23][:2].astype(np.int32)
            rh = keypoints[24][:2].astype(np.int32)
            torso_poly = np.array([ls, rs, rh, lh], dtype=np.int32)
            torso_roi  = np.zeros_like(warped_mask, dtype=np.uint8)
            cv2.fillConvexPoly(torso_roi, torso_poly, 255)
            torso_roi    = cv2.dilate(torso_roi, np.ones((21, 21), np.uint8), iterations=1)
            warped_mask  = cv2.bitwise_and(warped_mask.astype(np.uint8), torso_roi)
            warped_cloth = cv2.bitwise_and(warped_cloth, warped_cloth, mask=warped_mask)
        except Exception:
            pass

    agnostic_bgr = make_agnostic(person_bgr, pose_estimator, cloth_region_mask=warped_mask)

    # ── DEBUG GÖRSELLERİ ─────────────────────────────────────────────
    st.subheader("Debug Görseller")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.caption("Pose Map")
        st.image(cv2.cvtColor(pose_bgr, cv2.COLOR_BGR2RGB))
    with col2:
        st.caption("Cloth Mask")
        st.image(cloth_mask)
    with col3:
        st.caption("Warped Cloth")
        st.image(cv2.cvtColor(warped_cloth, cv2.COLOR_BGR2RGB))
    with col4:
        st.caption("Agnostic")
        st.image(cv2.cvtColor(agnostic_bgr, cv2.COLOR_BGR2RGB))
    # ─────────────────────────────────────────────────────────────────

    agnostic = norm_transform(Image.fromarray(cv2.cvtColor(agnostic_bgr, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
    cloth    = norm_transform(Image.fromarray(cv2.cvtColor(warped_cloth, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
    pose     = norm_transform(Image.fromarray(cv2.cvtColor(pose_bgr,     cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)

    cloth_mask_tensor = torch.from_numpy(warped_mask).float().unsqueeze(0).unsqueeze(0) / 255.0
    cloth_mask_tensor = cloth_mask_tensor.to(DEVICE)

    with torch.no_grad():
        output = generator(agnostic, cloth, cloth_mask_tensor, pose)

    person_rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)

    gen_rgb = np.array(tensor_to_image(output).convert("RGB"))

    # BUNU EKLE — GAN çıktısı yerine direkt warped cloth kullan
    gen_rgb = cv2.cvtColor(warped_cloth, cv2.COLOR_BGR2RGB)
    gen_rgb = cv2.resize(gen_rgb, (person_rgb.shape[1], person_rgb.shape[0]))
    

    torso_only_mask = cv2.erode(warped_mask, np.ones((3, 3), np.uint8), iterations=2)
    alpha = cv2.GaussianBlur(torso_only_mask, (21, 21), sigmaX=7).astype(np.float32) / 255.0
    alpha = np.expand_dims(alpha, 2)
    blended = (alpha * gen_rgb + (1 - alpha) * person_rgb).clip(0, 255).astype(np.uint8)

    r = blended[:, :, 0].astype(np.int16)
    g = blended[:, :, 1].astype(np.int16)
    b = blended[:, :, 2].astype(np.int16)
    green_artifact = (g > 115) & (g - r > 28) & (g - b > 20)
    blended[green_artifact] = person_rgb[green_artifact]

    return Image.fromarray(blended)


st.title("VTON Model Test UI")
st.write("Kisi ve kiyafet fotografi yukleyerek pose + segmentasyon + GAN zincirini test et.")

person_file = st.file_uploader("Kisi fotografi yukle", type=["jpg", "jpeg", "png"])
cloth_file = st.file_uploader("Urun / kiyafet fotografi yukle", type=["jpg", "jpeg", "png"])

if person_file and cloth_file:
    person_img = Image.open(person_file).convert("RGB")
    cloth_img = Image.open(cloth_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Kisi")
        st.image(person_img, width="stretch")
    with col2:
        st.subheader("Kiyafet")
        st.image(cloth_img, width="stretch")

    generator = load_generator()
    unet_model = load_unet()
    pose_estimator = load_pose_estimator()
    if not pose_estimator.available:
        st.warning("MediaPipe Pose mevcut degil. Pose haritasi bos tensor olarak kullanilacak.")

    if st.button("Modeli Calistir"):
        try:
            result = run_inference(
                person_img, cloth_img, generator, unet_model, pose_estimator
            )
            st.subheader("Sonuc")
            st.image(result, width="stretch")
        except Exception as e:
            st.error(f"Hata: {e}")
            import traceback
            st.code(traceback.format_exc())
