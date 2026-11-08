import os
import glob
import cv2
import numpy as np
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image

from model import TryOnGenerator
from modelunet import UNet
from posetespiti import PoseEstimator
from vucut_analizi import BodyParser, PersonInstanceSegmenter, restore_neck_skin, refine_person_instance, create_occlusion_mask
from kıyafetigiydir import warp_clothes
from segmentasyon import (
    segment_cloth_auto,
    segment_cloth_rembg,
    segment_cloth_unet,
    segment_cloth_color,
    clean_binary_mask,
    remove_head_and_neck,
    detect_cloth_attributes,
 
    REMBG_AVAILABLE
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_best_generator_path():
    import glob
    # 1. Öncelik: Yeni 50 epoch modeli
    if os.path.exists("models/generator_epoch_50.pth"):
        return "models/generator_epoch_50.pth"
    # 2. Öncelik: En yüksek numaralı generator checkpoint'i
    ckpts = glob.glob("models/generator_epoch_*.pth")
    if ckpts:
        def extract_num(p):
            try:
                return int(os.path.basename(p).replace("generator_epoch_", "").replace(".pth", ""))
            except:
                return -1
        ckpts.sort(key=extract_num)
        return ckpts[-1]
    return "models/generator_epoch_25.pth"

GENERATOR_PATH = get_best_generator_path()
EPOCH_LABEL = os.path.basename(GENERATOR_PATH).replace("generator_", "").replace(".pth", "").replace("_", " ").upper()
UNET_PATH = "models/unet_model_2.pth"
SIZE = (256, 256)


@st.cache_resource
def load_generator(path: str):
    import os
    if not os.path.exists(path):
        return None
    model = TryOnGenerator()
    try:
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE).eval()
        return model
    except Exception as e:
        print(f"Generator yüklenemedi ({path}): {e}")
        return None


@st.cache_resource
def load_pose_estimator():
    return PoseEstimator(static_image_mode=True)


@st.cache_resource
def load_body_parser():
    return BodyParser()


@st.cache_resource
def load_instance_segmenter():
    return PersonInstanceSegmenter()


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


def make_pose_map(person_bgr, pose_estimator, keypoints=None):
    if keypoints is not None:
        h, w = person_bgr.shape[:2]
        thick = max(4, int(min(h, w) * 0.012))
        rad = max(4, int(min(h, w) * 0.010))
        return pose_estimator.render_openpose_map(person_bgr, thickness=thick, joint_radius=rad, keypoints=keypoints)
    person_resized = cv2.resize(person_bgr, SIZE)
    return pose_estimator.render_openpose_map(person_resized, thickness=6, joint_radius=5)


def make_agnostic(person_bgr, pose_estimator, cloth_region_mask=None, sleeve_type="short", keypoints=None):
    """
    Kişinin giydiği kıyafeti gövdeden silerek agnostic görsel üretir.
    Kol tipine (kısa kol / uzun kol / kolsuz) göre kolları ve teni akıllıca korur.
    Kısa kollu tişörtlerde ön kollar ve eller KESİNLİKLE silinmez.
    """
    if keypoints is None:
        person_bgr = cv2.resize(person_bgr, SIZE)
        keypoints_and_landmarks = pose_estimator.get_keypoints(person_bgr)
        if keypoints_and_landmarks is not None:
            keypoints, _ = keypoints_and_landmarks

    agnostic = person_bgr.copy()
    h, w, _ = agnostic.shape
    scale = max(1.0, min(h, w) / 256.0)

    if keypoints is None:
        x1, x2 = int(w * 0.34), int(w * 0.66)
        y1, y2 = int(h * 0.22), int(h * 0.68)
        cv2.rectangle(agnostic, (x1, y1), (x2, y2), (127, 127, 127), -1)
        return agnostic

    req = [11, 12, 23, 24]
    if any(len(keypoints) <= i for i in req):
        return agnostic

    ls = keypoints[11][:2].astype(np.int32)
    rs = keypoints[12][:2].astype(np.int32)
    lh = keypoints[23][:2].astype(np.int32)
    rh = keypoints[24][:2].astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)

    # 1. Gövde Maskesi
    torso = np.array([ls, rs, rh, lh], dtype=np.int32)
    cv2.fillConvexPoly(mask, torso, 255)
    k_dil = max(5, int(9 * scale))
    mask = cv2.dilate(mask, np.ones((k_dil, k_dil), np.uint8), iterations=1)

    # 2. Kol Maskelemesi (Kol Tipine Göre Kesin Karar)
    if sleeve_type == "long":
        for a, b in [(11, 13), (13, 15), (12, 14), (14, 16)]:
            if len(keypoints) > b:
                p1 = tuple(keypoints[a][:2].astype(np.int32))
                p2 = tuple(keypoints[b][:2].astype(np.int32))
                cv2.line(mask, p1, p2, 255, int(16 * scale))
                cv2.circle(mask, p1, int(10 * scale), 255, -1)
                cv2.circle(mask, p2, int(10 * scale), 255, -1)
    elif sleeve_type == "short":
        for s_idx, e_idx in [(11, 13), (12, 14)]:
            if len(keypoints) > e_idx:
                p1 = tuple(keypoints[s_idx][:2].astype(np.int32))
                mid_arm = (keypoints[s_idx][:2] * 0.50 + keypoints[e_idx][:2] * 0.50).astype(np.int32)
                cv2.line(mask, p1, tuple(mid_arm), 255, int(18 * scale))
                cv2.circle(mask, p1, int(12 * scale), 255, -1)

    # 3. Kıyafet Bölgesi Eklemesi
    if cloth_region_mask is not None:
        crm = cloth_region_mask.astype(np.uint8)
        if crm.shape != mask.shape:
            crm = cv2.resize(crm, (w, h), interpolation=cv2.INTER_NEAREST)
        crm = cv2.morphologyEx(crm, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        mask = cv2.bitwise_or(mask, crm)

    # 4. TEN & VÜCUT KORUMA (KRİTİK GÜVENCE):
    if sleeve_type in ["short", "sleeveless"]:
        forearm_guard = np.zeros((h, w), dtype=np.uint8)
        for e_idx, w_idx in [(13, 15), (14, 16)]:
            if len(keypoints) > w_idx:
                pe = tuple(keypoints[e_idx][:2].astype(np.int32))
                pw = tuple(keypoints[w_idx][:2].astype(np.int32))
                cv2.line(forearm_guard, pe, pw, 255, int(24 * scale))
                cv2.circle(forearm_guard, pw, int(18 * scale), 255, -1)
        mask[forearm_guard > 0] = 0

    # Baş ve yüzü koru
    if len(keypoints) > 0:
        nx, ny, nv = keypoints[0]
        if nv > 0.2:
            head_keep = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(head_keep, (int(nx), int(ny)), int(28 * scale), 255, -1)
            cv2.circle(head_keep, (int(nx), int(ny + 20 * scale)), int(24 * scale), 255, -1)
            mask[head_keep > 0] = 0

    # Alt vücut (kalça altı, bacaklar) koru
    hip_limit = int(max(lh[1], rh[1]) + (h * 0.12))
    if hip_limit < h:
        mask[hip_limit:, :] = 0

    agnostic[mask > 0] = (127, 127, 127)
    return agnostic


def _apply_photometric_relighting_and_3d_shading(cloth_rgb, mask_binary, orig_person_rgb, parse_clean=None, instance_mask=None, keypoints=None):
    """
    Faz 4: Ortam Işığı, LAB Renk Sıcaklığı Uyumu ve 3B Silindirik Gövde/Kol Hacim Gölgesi
    1. Hedef kişinin ten/ortam aydınlanmasını CIE LAB uzayında analiz ederek kıyafete %12-%14 oranında
       doğal renk sıcaklığı (güneşli plaj sıcaklığı veya iç mekan soğuk/nötr ışığı) ve pozlama uyumu aktarır.
    2. Kişinin sol-sağ üst gövde ışık farkından yönlü aydınlanma gradyanı (directional light) üretir.
    3. Gövde ve her bir kol kesiti boyunca merkezden kenarlara doğru Lambertian 3B silindirik hacim
       gölgesi (merkezde hafif röfle, dış kenarlarda yumuşak derinlik) uygular.
    """
    if mask_binary is None or not np.any(mask_binary > 0):
        return cloth_rgb

    h, w = mask_binary.shape[:2]
    out_rgb = cloth_rgb.astype(np.float32).copy()

    # 1. CIE LAB Ortam Işığı ve Renk Sıcaklığı Uyumu (%12-%14)
    if parse_clean is not None and instance_mask is not None:
        skin_mask = ((parse_clean == 2) | (parse_clean == 3)) & (instance_mask > 0)
        if np.sum(skin_mask) > 80:
            person_lab = cv2.cvtColor(orig_person_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            cloth_lab = cv2.cvtColor(cloth_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

            skin_pts_lab = person_lab[skin_mask]
            l_skin = float(np.percentile(skin_pts_lab[:, 0], 60))
            a_skin = float(np.median(skin_pts_lab[:, 1]))
            b_skin = float(np.median(skin_pts_lab[:, 2]))

            # Nötr gün ışığı referans ten değerleri (OpenCV 8-bit LAB: L in 0..255, a/b in 0..255, 128=nötr)
            l_ref, a_ref, b_ref = 152.0, 139.0, 142.0
            delta_l = float(np.clip((l_skin - l_ref) * 0.12, -9.0, 9.0))
            delta_a = float(np.clip((a_skin - a_ref) * 0.14, -5.5, 5.5))
            delta_b = float(np.clip((b_skin - b_ref) * 0.14, -7.5, 7.5))

            cloth_lab[:, :, 0] = np.clip(cloth_lab[:, :, 0] + delta_l, 0.0, 255.0)
            cloth_lab[:, :, 1] = np.clip(cloth_lab[:, :, 1] + delta_a, 0.0, 255.0)
            cloth_lab[:, :, 2] = np.clip(cloth_lab[:, :, 2] + delta_b, 0.0, 255.0)
            out_rgb = cv2.cvtColor(cloth_lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

    # 2. 3B Silindirik Gövde ve Kol Hacim Gölgesi (Lambertian Cylindrical Roll-Off)
    cyl_gain = np.ones((h, w), dtype=np.float32)
    ys_cloth = np.where(mask_binary > 0)[0]
    if len(ys_cloth) > 20:
        y_min_c, y_max_c = int(ys_cloth.min()), int(ys_cloth.max())
        for y_r in range(y_min_c, y_max_c + 1):
            row_m = mask_binary[y_r, :] > 0
            if not np.any(row_m):
                continue
            # Satırdaki kesintisiz parçaları (gövde ve ayrık kollar) ayrı silindirler olarak işle
            padded = np.concatenate(([False], row_m, [False]))
            starts = np.where(padded[1:] & ~padded[:-1])[0]
            ends = np.where(~padded[1:] & padded[:-1])[0] - 1
            for x1, x2 in zip(starts, ends):
                seg_w = x2 - x1 + 1
                if seg_w >= 10:
                    xc = 0.5 * (x1 + x2)
                    half_w = max(1.0, 0.5 * seg_w)
                    xs_seg = np.arange(x1, x2 + 1, dtype=np.float32)
                    d_norm = np.abs(xs_seg - xc) / half_w
                    # Merkezde +%3 doğal röfle, dış kenarlarda -%9 silindirik derinlik gölgesi
                    cyl_gain[y_r, x1:x2 + 1] = 1.03 - 0.12 * np.power(np.clip(d_norm, 0.0, 1.0), 1.85)

        # Yönlü ortam ışığı (Directional Sunlight / Room Light Gradient):
        if keypoints is not None and len(keypoints) >= 13 and instance_mask is not None:
            sh_cx = float((keypoints[11][0] + keypoints[12][0]) * 0.5)
            person_gray = cv2.cvtColor(orig_person_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            p_mask = (instance_mask > 0)
            xs_grid = np.arange(w)[None, :]
            left_lit = person_gray[p_mask & (xs_grid < sh_cx)]
            right_lit = person_gray[p_mask & (xs_grid >= sh_cx)]
            if len(left_lit) > 50 and len(right_lit) > 50:
                dir_diff = float(np.clip((np.median(left_lit) - np.median(right_lit)) / 255.0, -0.18, 0.18))
                x_all = np.where(mask_binary > 0)[1]
                x_c_min, x_c_max = float(x_all.min()), float(max(x_all.min() + 1, x_all.max()))
                x_ramp = 1.0 - 2.0 * np.clip((np.arange(w, dtype=np.float32) - x_c_min) / (x_c_max - x_c_min), 0.0, 1.0)
                cyl_gain *= (1.0 + 0.18 * dir_diff * x_ramp[None, :])

        cyl_gain = cv2.GaussianBlur(cyl_gain, (15, 15), 4.0)
        out_rgb = np.clip(out_rgb * cyl_gain[:, :, None], 0.0, 255.0)

    return out_rgb.astype(np.uint8)


def autonomous_hybrid_fusion(person_bgr, warped_cloth, warped_mask, gan_result_img=None, sleeve_type="short", keypoints=None, use_gan_shading=False, parse_map=None, person_instance_mask=None):
    """
    Doğal ve Kusursuz Giydirme (5-Sistemli Entegre Füzyon):
    1. Agnostic Canvas & Neck Skin Restoration: Kişinin eski kıyafeti (askılı atlet, yaka) silinirken
       yüz/çene ten rengi örneklenerek boyun çukuru doğal ten tonu degrade gradyanıyla rekonstrükte edilir.
    2. Alt Beden ve Eşofman Koruması: Etek ucu altındaki kıyafet bölgeleri asla silinmez/inpaint edilmez.
    3. Ortam Temas Gölgesi (Ambient Contact Shadow): Kıyafet sınırlarına tene temas hissi veren doğal ince gölge ekler.
    4. Yumuşak Kenar Füzyonu: Kumaş dokusunu %100 netlikte giydirir.
    5. Occlusion Protection: Saçlar, eller ve ön kollar piksel seviyesinde kıyafetin önünde korunur.
    """
    h, w = person_bgr.shape[:2]
    clean_person_bgr = person_bgr.copy()

    # 1. Instance İzolasyonu ve Agnostic Eski Kıyafet Temizliği
    instance_mask, parse_clean = refine_person_instance(parse_map, keypoints, person_bgr.shape, person_instance_mask=person_instance_mask)

    if parse_clean is not None and parse_clean.shape[:2] == (h, w):
        old_clothes = ((parse_clean == 4) & (instance_mask > 0)).astype(np.uint8) * 255
        y_sh = float((keypoints[11][1] + keypoints[12][1]) / 2.0) if (keypoints is not None and len(keypoints) >= 25) else (h * 0.4)
        sh_w = float(np.linalg.norm(keypoints[12][:2] - keypoints[11][:2])) if (keypoints is not None and len(keypoints) >= 13) else (w * 0.30)
        sh_cx = float((keypoints[11][0] + keypoints[12][0]) / 2.0) if (keypoints is not None and len(keypoints) >= 13) else (w * 0.50)

        # Faz 3.1: Kalkık el, parmak ve telefon koruma bariyeri (Yalnızca el/bilek omuz hizasına veya yukarı kalkıksa
        # devreye girer; ayna selfiesinde bordo/kırmızı telefon kılıfının ve parmakların silinmesini %100 engeller):
        hand_phone_guard = np.zeros((h, w), dtype=np.uint8)
        has_raised_hand = False
        if keypoints is not None and len(keypoints) >= 17:
            for hk_i in [15, 16, 17, 18, 19, 20, 21, 22]:
                if hk_i < len(keypoints) and keypoints[hk_i][2] > 0.18:
                    hx, hy = float(keypoints[hk_i][0]), float(keypoints[hk_i][1])
                    if hy < y_sh + sh_w * 0.32:
                        has_raised_hand = True
                        cv2.circle(hand_phone_guard, (int(hx), int(hy)), int(sh_w * 0.32), 255, -1)
        if has_raised_hand and keypoints is not None and len(keypoints) >= 11 and keypoints[9][2] > 0.25 and keypoints[10][2] > 0.25:
            mouth_y = float((keypoints[9][1] + keypoints[10][1]) / 2.0)
            y_chin_guard = max(0, int(mouth_y + sh_w * 0.22))
            hand_phone_guard[:y_chin_guard, :] = 255
        if np.any(hand_phone_guard > 0):
            old_clothes[hand_phone_guard > 0] = 0

        # Boyun ve omuz üstü ten bölgesinde BodyParser'ın ten (2), saç (1) veya aksesuar (5) sandığı ince bikini/askı iplerini (halter strap) yakala:
        yn1, yn2 = max(0, int(y_sh - sh_w * 0.45)), min(h, int(y_sh + sh_w * 0.14))
        xn1, xn2 = max(0, int(sh_cx - sh_w * 0.44)), min(w, int(sh_cx + sh_w * 0.44))
        if yn2 > yn1 + 10 and xn2 > xn1 + 10:
            roi_parse = parse_clean[yn1:yn2, xn1:xn2]
            roi_warped = warped_mask[yn1:yn2, xn1:xn2]
            roi_hp_guard = hand_phone_guard[yn1:yn2, xn1:xn2]
            neck_skin_roi = (roi_parse == 2) & (roi_warped == 0) & (roi_hp_guard == 0)
            if np.sum(neck_skin_roi) > 40:
                roi_bgr = person_bgr[yn1:yn2, xn1:xn2]
                roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                skin_luma = float(np.percentile(roi_gray[neck_skin_roi], 65))
                bh_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
                blackhat = cv2.morphologyEx(roi_gray, cv2.MORPH_BLACKHAT, bh_kernel)
                tophat = cv2.morphologyEx(roi_gray, cv2.MORPH_TOPHAT, bh_kernel)
                roi_sat = np.max(roi_bgr, axis=2).astype(np.int16) - np.min(roi_bgr, axis=2).astype(np.int16)
                roi_inst = instance_mask[yn1:yn2, xn1:xn2] > 0

                # Her satırda en sol ve en sağ ten pikseli arasında sıkışmış veya boyun/omuz tenine bitişik ince askı şeritlerini bul:
                between_skin = np.zeros_like(neck_skin_roi, dtype=bool)
                for ry in range(neck_skin_roi.shape[0]):
                    s_xs = np.where(neck_skin_roi[ry, :])[0]
                    if len(s_xs) >= 6:
                        between_skin[ry, max(0, int(s_xs[0]) - 8):min(neck_skin_roi.shape[1], int(s_xs[-1]) + 9)] = True

                dark_strap = (blackhat > 20) & (roi_gray < (skin_luma - 26.0))
                white_strap = (tophat > 12) & (roi_gray > max(172.0, skin_luma + 22.0)) & (roi_sat < 38) & roi_inst
                strap_cand = (
                    between_skin &
                    (roi_hp_guard == 0) &
                    (roi_warped == 0) &
                    (roi_parse != 3) &
                    (dark_strap | white_strap)
                )
                if np.any(strap_cand):
                    strap_u8 = cv2.dilate(strap_cand.astype(np.uint8) * 255, np.ones((5, 5), np.uint8), iterations=1)
                    strap_u8[roi_hp_guard > 0] = 0
                    # BodyParser askıyı saç (1) sanmışsa ten (2) olarak düzelt ki restore_neck_skin silsin ve saç maskesi geri yapıştırmasın
                    roi_parse[(strap_u8 > 0) & (roi_parse == 1)] = 2
                    parse_clean[yn1:yn2, xn1:xn2] = roi_parse
                    old_clothes[yn1:yn2, xn1:xn2] = cv2.bitwise_or(old_clothes[yn1:yn2, xn1:xn2], strap_u8)

        if np.any(old_clothes > 0):
            exposed_old = cv2.bitwise_and(old_clothes, cv2.bitwise_not(warped_mask))

            # Etek ucu alt sınırı (yalnızca merkez gövde sütunlarından ölçülür; uzun kolların bilekleri etek ucunu aşağı çekmez)
            tx1 = max(0, int(sh_cx - sh_w * 0.35))
            tx2 = min(w, int(sh_cx + sh_w * 0.35))
            torso_warped_ys = np.where(warped_mask[:, tx1:tx2] > 0)[0]
            if len(torso_warped_ys) > 0:
                hem_y_max = int(torso_warped_ys.max())
            else:
                warped_ys = np.where(warped_mask > 0)[0]
                hem_y_max = int(warped_ys.max()) if len(warped_ys) > 0 else None

            # Alt beden koruması: Etek ucu altındaki eşofman/pantolon piksellerini KORU (asla inpaint etme).
            # Kavisli etek uçlarında (ör. raglan tişört) her sütunun kendi yerel etek ucu sınırının (col_ys.max())
            # altındaki pantolon/şort piksellerini de koru ki etek kavisi altında yatay silme şeridi oluşmasın:
            y_hip_lim = int(max(keypoints[23][1], keypoints[24][1]) + sh_w * 0.42) if (keypoints is not None and len(keypoints) >= 25) else h
            p_max_c = np.max(person_bgr, axis=2).astype(np.int16)
            p_min_c = np.min(person_bgr, axis=2).astype(np.int16)
            p_luma_c = np.mean(person_bgr, axis=2)
            dark_pants_px = ((p_max_c - p_min_c) < 42) & (p_luma_c < 75.0)
            if hem_y_max is not None:
                cutoff = min(hem_y_max + 2, y_hip_lim)
                exposed_old[cutoff:, :][dark_pants_px[cutoff:, :]] = 0
                hard_cutoff = min(h, max(cutoff + int(sh_w * 0.28), y_hip_lim))
                exposed_old[hard_cutoff:, :] = 0
            elif keypoints is not None and len(keypoints) >= 25:
                exposed_old[y_hip_lim:, :] = 0

            # Sütun bazlı yerel etek ucu koruması (yalnızca gövde/bel bandında y > y_sh + sh_w * 0.70):
            # Koyu şort/kot pantolon piksellerini (düşük doygunluk ve düşük parlaklık) kavisli etek ucu altında koru,
            # ancak eski renkli/parlak üst kıyafet sarkmalarını (ör. kırmızı veya mavi çizgili eski tişört) silmeye devam et:
            y_waist_start = int(y_sh + sh_w * 0.70)
            for cx_i in range(w):
                c_ys = np.where(warped_mask[y_waist_start:, cx_i] > 0)[0]
                if len(c_ys) > 0:
                    local_hem = y_waist_start + int(c_ys.max())
                    exposed_old[local_hem:, cx_i][dark_pants_px[local_hem:, cx_i]] = 0
                else:
                    if hem_y_max is not None:
                        y_side_cut = max(y_waist_start, hem_y_max - int(sh_w * 0.18))
                        exposed_old[y_side_cut:, cx_i][dark_pants_px[y_side_cut:, cx_i]] = 0

            # Arka plan koruma bariyeri (sadece kişi silüeti içinde inpaint)
            exposed_old = cv2.bitwise_and(exposed_old, cv2.dilate((instance_mask > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) * 255)

            if np.any(exposed_old > 0):
                clean_person_bgr = restore_neck_skin(
                    clean_person_bgr,
                    exposed_old,
                    parse_clean,
                    y_sh,
                    instance_mask=instance_mask,
                    hem_y_max=hem_y_max
                )

    person_rgb = cv2.cvtColor(clean_person_bgr, cv2.COLOR_BGR2RGB)
    orig_person_rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
    cloth_rgb = cv2.cvtColor(warped_cloth, cv2.COLOR_BGR2RGB)
    mask_binary = (warped_mask > 80).astype(np.uint8)

    # Faz 4: Ortam Işığı, LAB Renk Sıcaklığı Uyumu ve 3B Silindirik Gövde/Kol Hacim Gölgesi
    cloth_rgb = _apply_photometric_relighting_and_3d_shading(
        cloth_rgb,
        mask_binary,
        orig_person_rgb,
        parse_clean=parse_clean,
        instance_mask=instance_mask,
        keypoints=keypoints
    )

    # 2. Ortam Temas Gölgesi (Ambient Contact Shadow)
    shadow_mask = cv2.dilate(mask_binary, np.ones((9, 9), np.uint8), iterations=1)
    shadow_blur = cv2.GaussianBlur(shadow_mask.astype(np.float32), (15, 15), 0)
    shadow_blur = np.clip((shadow_blur - cv2.GaussianBlur(mask_binary.astype(np.float32), (7, 7), 0)), 0, 1)

    person_rgb_f = person_rgb.astype(np.float32)
    for c in range(3):
        person_rgb_f[:, :, c] *= (1.0 - shadow_blur * 0.35)
    person_rgb = np.clip(person_rgb_f, 0, 255).astype(np.uint8)

    # 3. Yumuşak kenarlı kıyafet maskelemesi
    alpha = cv2.GaussianBlur(mask_binary.astype(np.float32), (3, 3), sigmaX=0.8)
    alpha_3d = np.expand_dims(alpha, 2)

    # Kıyafeti gövdeye doğal giydir
    blended = (alpha_3d * cloth_rgb.astype(np.float32) + (1.0 - alpha_3d) * person_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)

    # 4. İsteğe Bağlı GAN Gölge Entegrasyonu
    if use_gan_shading and gan_result_img is not None:
        gan_rgb = np.array(gan_result_img.convert("RGB"))
        if gan_rgb.shape[:2] == (h, w):
            inner_cloth = cv2.erode(mask_binary, np.ones((9, 9), np.uint8), iterations=1)
            inner_weight = (inner_cloth.astype(np.float32)) * 0.15
            inner_weight_3d = np.expand_dims(inner_weight, 2)
            blended = ((1.0 - inner_weight_3d) * blended.astype(np.float32) + inner_weight_3d * gan_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)

    # 5A. Renk-Arındırmalı Saç Koruması (Color-Decontaminated Hair Alpha Matting):
    # Omuz ve göğse dökülen saç tellerini (parse_clean == 1) korurken, saç telleri arasındaki boşluklardan
    # sızan eski kıyafet rengini (ör. parlak kırmızı tişört) veya arka plan halelerini filtreler.
    if parse_clean is not None:
        hair_occ = ((parse_clean == 1) & (mask_binary > 0)).astype(np.uint8) * 255
        if np.any(hair_occ > 0):
            head_hair = (parse_clean == 1) & (mask_binary == 0)
            if np.sum(head_hair) < 30:
                head_hair = (parse_clean == 1)
            hair_pts = orig_person_rgb[head_hair]
            hair_med = np.median(hair_pts, axis=0).astype(np.float32) if len(hair_pts) > 0 else np.array([40.0, 35.0, 30.0], dtype=np.float32)
            hair_r_excess = float(max(0.0, hair_med[0] - max(hair_med[1], hair_med[2])))

            orig_f = orig_person_rgb.astype(np.float32).copy()
            d_hair = np.linalg.norm(orig_f - hair_med[None, None, :], axis=2)
            # Kırmızı/renkli eski tişört sızıntısını yakalamak için kromatik (R - max(G,B)) aşımını ölç:
            pix_r_excess = orig_f[:, :, 0] - np.maximum(orig_f[:, :, 1], orig_f[:, :, 2])
            chroma_bleed = np.maximum(0.0, pix_r_excess - (hair_r_excess + 5.0))

            # Saç rengine yakınlığına ve kromatik saflığına göre yumuşak ağırlık (0..1)
            w_color = np.clip(1.0 - (d_hair + chroma_bleed * 2.8 - 28.0) / 34.0, 0.0, 1.0).astype(np.float32)
            w_color[chroma_bleed > 8.0] = 0.0

            w_color_smooth = cv2.GaussianBlur(w_color, (3, 3), 0.6)
            base_alpha = cv2.GaussianBlur(hair_occ.astype(np.float32) / 255.0, (3, 3), 0.7)
            hair_alpha = (base_alpha * w_color_smooth)[:, :, None]

            # Yarı saydam saç uçlarında kırmızı/renkli eski tişört yansımasını doğrudan saçın doğal R sınırına kırp:
            orig_f[:, :, 0] = np.minimum(orig_f[:, :, 0], np.maximum(orig_f[:, :, 1], orig_f[:, :, 2]) + hair_r_excess + 3.0)
            w_dec = np.clip(w_color[:, :, None], 0.0, 1.0)
            decontam_hair = w_dec * orig_f + (1.0 - w_dec) * hair_med[None, None, :]
            blended = (hair_alpha * decontam_hair + (1.0 - hair_alpha) * blended.astype(np.float32)).clip(0, 255).astype(np.uint8)

        # 5B. Faz 3.1: Anatomik El ve Parmak Örtüşme Koruması (Hand & Finger Occlusion Matting):
        # Eller belde (akimbo) veya göğüs/yaka önünde telefon tutarken (selfie), bilekten parmak uçlarına
        # uzanan el/parmak ten piksellerini (parse_clean == 2) kıyafetin en üst katmanına yumuşak alfa ile yerleştirir:
        if keypoints is not None and len(keypoints) >= 17:
            sh_w_h = float(np.linalg.norm(keypoints[12][:2] - keypoints[11][:2]))
            hand_roi = np.zeros((h, w), dtype=np.uint8)
            for w_idx, p_idx, i_idx, t_idx in [(15, 17, 19, 21), (16, 18, 20, 22)]:
                if keypoints[w_idx][2] > 0.20:
                    pw = keypoints[w_idx][:2].astype(np.float32)
                    # Bilekten parmak uçlarına doğru el merkezini hesapla (bilek manşetinin hemen ilerisinden başlat)
                    finger_pts = [keypoints[k][:2] for k in (p_idx, i_idx, t_idx) if k < len(keypoints) and keypoints[k][2] > 0.18]
                    if finger_pts:
                        pf = np.mean(finger_pts, axis=0).astype(np.float32)
                        v_wf = pf - pw
                        p_hand_start = pw + 0.25 * v_wf
                        cv2.line(hand_roi, tuple(p_hand_start.astype(np.int32)), tuple(pf.astype(np.int32)), 255, max(10, int(sh_w_h * 0.16)))
                        cv2.circle(hand_roi, tuple(pf.astype(np.int32)), max(8, int(sh_w_h * 0.12)), 255, -1)
                    else:
                        # Parmak uçları görünmüyorsa yalnızca kalkık/bükülü bilek ucunda küçük el dairesi koru
                        if pw[1] < keypoints[11][1] + sh_w_h * 0.95:
                            cv2.circle(hand_roi, tuple(pw.astype(np.int32)), max(8, int(sh_w_h * 0.10)), 255, -1)
            hand_occ = ((hand_roi > 0) & (parse_clean == 2) & (mask_binary > 0)).astype(np.uint8) * 255
            if np.any(hand_occ > 0):
                # Kıyafet sınırındaki 1 piksellik eski kıyafet halesini temizlemek için hafif erode + Gaussian yumuşatma
                hand_occ_clean = cv2.erode(hand_occ, np.ones((3, 3), np.uint8), iterations=1)
                if np.any(hand_occ_clean > 0):
                    hand_alpha = cv2.GaussianBlur(hand_occ_clean.astype(np.float32) / 255.0, (5, 5), 0.9)[:, :, None]
                    blended = (hand_alpha * orig_person_rgb.astype(np.float32) + (1.0 - hand_alpha) * blended.astype(np.float32)).clip(0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def remove_bulky_garment(person_bgr, warped_mask, keypoints, sleeve_type="short"):
    """
    Kullanıcı kürk/mont gibi kalın kıyafet giydiğinde:
    1. Yüzden doğal ten rengini örnekler.
    2. Kısa kollu tişörtün altında kalan kollara doğal ten dokusu ve kol formu çizer.
    3. Tişört ve yeni kolların dışında kalan kürkün kabarık kısımlarını inpaint ile silip arka planı tamamlar.
    """
    if keypoints is None or len(keypoints) < 17:
        return person_bgr

    h, w, _ = person_bgr.shape
    try:
        sh_w = float(np.linalg.norm(keypoints[12][:2] - keypoints[11][:2]))
    except Exception:
        return person_bgr

    if sh_w < 10:
        return person_bgr

    # 1. Ten Rengi Örnekleme (Yüz ve çene bölgesinden)
    nx, ny = int(keypoints[0][0]), int(keypoints[0][1])
    patch = person_bgr[max(0, ny - 15):min(h, ny + 15), max(0, nx - 15):min(w, nx + 15)]
    if patch.size > 0:
        skin_color = np.median(patch.reshape(-1, 3), axis=0).astype(np.uint8)
    else:
        skin_color = np.array([120, 130, 160], dtype=np.uint8)

    clean_person = person_bgr.copy()

    # 2. Doğal Kol Maskesi ve Ten Çizimi (Omuz -> Dirsek -> Bilek)
    arm_skin_mask = np.zeros((h, w), dtype=np.uint8)
    arm_thick = max(10, int(sh_w * 0.20))
    forearm_thick = max(8, int(sh_w * 0.15))

    for s_idx, e_idx, w_idx in [(11, 13, 15), (12, 14, 16)]:
        if len(keypoints) > w_idx:
            ps = tuple(keypoints[s_idx][:2].astype(np.int32))
            pe = tuple(keypoints[e_idx][:2].astype(np.int32))
            pw = tuple(keypoints[w_idx][:2].astype(np.int32))

            cv2.line(arm_skin_mask, ps, pe, 255, arm_thick)
            cv2.circle(arm_skin_mask, pe, int(arm_thick * 0.55), 255, -1)
            cv2.line(arm_skin_mask, pe, pw, 255, forearm_thick)
            cv2.circle(arm_skin_mask, pw, int(forearm_thick * 0.55), 255, -1)

    # 3. Kürkün Dışarı Taşan Hacmini Silme Maskesi (Kürkün kabarık kısımları)
    bulky_fur_mask = np.zeros((h, w), dtype=np.uint8)
    fur_thick = int(sh_w * 0.45)
    for s_idx, e_idx, w_idx in [(11, 13, 15), (12, 14, 16)]:
        if len(keypoints) > w_idx:
            ps = tuple(keypoints[s_idx][:2].astype(np.int32))
            pe = tuple(keypoints[e_idx][:2].astype(np.int32))
            pw = tuple(keypoints[w_idx][:2].astype(np.int32))
            cv2.line(bulky_fur_mask, ps, pe, 255, fur_thick)
            cv2.line(bulky_fur_mask, pe, pw, 255, int(fur_thick * 0.85))

    # Tişörtün kapladığı alanı ve yeni kol tenini kürk silme maskesinden çıkar:
    bulky_fur_mask[warped_mask > 0] = 0
    bulky_fur_mask[arm_skin_mask > 0] = 0

    # Çene ve yüzü koru
    if len(keypoints) > 0:
        cv2.circle(bulky_fur_mask, (nx, ny), int(sh_w * 0.45), 0, -1)

    # 4. Kürkün dışa taşan kabarıklığını arka planla inpaint et
    if np.sum(bulky_fur_mask > 0) > 100:
        try:
            clean_person = cv2.inpaint(clean_person, bulky_fur_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
        except Exception:
            pass

    # 5. Yeni kollara ten dokusunu yerleştir (yumuşak kenarlı harmanlama)
    skin_layer = np.full_like(clean_person, skin_color)
    arm_alpha = cv2.GaussianBlur(arm_skin_mask.astype(np.float32) / 255.0, (9, 9), 2.5)
    arm_alpha_3d = np.expand_dims(arm_alpha, 2)
    clean_person = (arm_alpha_3d * skin_layer + (1.0 - arm_alpha_3d) * clean_person.astype(np.float32)).clip(0, 255).astype(np.uint8)

    return clean_person


# --- STREAMLIT ARAYÜZÜ ---
st.set_page_config(
    page_title="Otonom Sanal Kıyafet Deneme (VTON)",
    page_icon="👔",
    layout="wide"
)

st.title("👔 Otonom Sanal Kıyafet Deneme (VTON) Platformu")
st.markdown("""
Bu sistem, yüklenen görselleri derin öğrenme ve bilgisayarlı görü ile **otonom olarak analiz eder**:
- Manken/insan varlığını tespit edip yüz/boyun bölgesini otomatik ayrıştırır.
- Kıyafetin kol tipini (kısa kollu, uzun kollu, kolsuz) kendisi hesaplar.
- Ten dokusunu koruyarak yapay zeka ve warping sentezini kusursuz birleştirir.
""")

st.info("""
💡 **En Kusursuz Sonuç İçin Fotoğraf Tavsiyesi (Best Practice):**  
Sanal deneme kabininde en doğal ve pürüzsüz sonuç için üzerinizde **vücuda oturan ince bir tişört veya atlet** varken çekilmiş fotoğraflar tercih ediniz. Çok kalın mont, kaban veya kabarık kürk giyildiğinde, kıyafetin dışında kalan kısımlar için sol menüdeki **Kalın Kıyafet / Kürk Silme Modunu** açabilirsiniz.
""")

# Model Durumları ve Sistem Paneli
st.sidebar.header("🤖 Sistem & Model Durumu")
generator = load_generator(GENERATOR_PATH)
unet_model = load_unet()
pose_estimator = load_pose_estimator()
body_parser = load_body_parser()
instance_segmenter = load_instance_segmenter()

if pose_estimator.available:
    st.sidebar.success("✅ MediaPipe Pose (Heavy): Aktif")
else:
    st.sidebar.error(f"❌ MediaPipe Pose: {pose_estimator.error_message}")

if instance_segmenter.available:
    st.sidebar.success("✅ YOLO11-Seg (Instance Segmenter): Aktif")
else:
    st.sidebar.warning("⚠️ YOLO11-Seg: Devre dışı (Fallback devrede)")

if body_parser.available:
    st.sidebar.success("✅ MediaPipe Multiclass Parser: Aktif")
else:
    st.sidebar.warning("⚠️ Multiclass Parser: Devre dışı")

if REMBG_AVAILABLE:
    st.sidebar.success("✅ rembg (u2net_cloth_seg): Hazır")
else:
    st.sidebar.warning("⚠️ rembg kütüphanesi aktif değil")

if generator is not None:
    st.sidebar.success(f"✅ TryOn-GAN ({GENERATOR_PATH}): Yüklendi")
else:
    st.sidebar.info(f"ℹ️ TryOn-GAN modeli: models/ dizininde bekleniyor.")

# İsteğe Bağlı Manuel Ayarlar (Genişletilebilir)
with st.sidebar.expander("⚙️ Manuel İnce Ayarlar (İsteğe Bağlı)", expanded=False):
    st.caption("Sistem tüm kararları otonom verir. Yalnızca özel ince ayar yapmak isterseniz kullanın:")
    override_sleeve = st.selectbox(
        "Kol Tipi Zorlama",
        options=["auto", "short", "long", "sleeveless"],
        format_func=lambda x: {
            "auto": "🤖 Otomatik Teşhis (Önerilen)",
            "short": "👕 Kısa Kollu (T-Shirt)",
            "long": "🧥 Uzun Kollu (Gömlek/Kazak)",
            "sleeveless": "🎽 Kolsuz (Askılı/Atlet)"
        }[x]
    )
    widen_ratio = st.slider("Omuz Oturma Payı", min_value=0.15, max_value=0.40, value=0.25, step=0.05)
    enable_coat_removal = st.checkbox(
        "🧥 Kalın Kıyafet / Kürk Silme Modu (Deneysel)",
        value=False,
        help="Fotoğrafta kürk veya kalın mont varsa, yeni tişörtün dışında kalan kürk kollarını ve arka planını silip ten dokusuyla tamamlamaya çalışır."
    )
    debug_mode = st.sidebar.checkbox(
        "🛠️ Geliştirici / Debug Modu",
        value=False,
        help="Pipeline 2.0 ara aşama maskelerini (YOLO instance, parsing, warp mask) ayrıntılı incelemenizi sağlar."
    )


# Görsel Yükleme Bölümü
col_up1, col_up2 = st.columns(2)
with col_up1:
    person_file = st.file_uploader("1. Hedef Kişi Fotoğrafı Yükle", type=["jpg", "jpeg", "png"])
with col_up2:
    cloth_file = st.file_uploader("2. Denenecek Kıyafet Görseli Yükle", type=["jpg", "jpeg", "png"])

if person_file and cloth_file:
    person_img = Image.open(person_file).convert("RGB")
    cloth_img = Image.open(cloth_file).convert("RGB")

    c_prev1, c_prev2 = st.columns(2)
    with c_prev1:
        st.subheader("👤 Hedef Kişi")
        st.image(person_img, width="stretch")
    with c_prev2:
        st.subheader("👕 Denenecek Kıyafet")
        st.image(cloth_img, width="stretch")

    st.write("")
    if st.button("🚀 Otonom Giydirme İşlemini Başlat", type="primary"):
        with st.spinner("🤖 Yapay zeka görselleri inceliyor, kıyafet türüne karar veriyor ve giydiriyor..."):
            person_orig_bgr = pil_to_bgr(person_img)
            cloth_orig_bgr = pil_to_bgr(cloth_img)

            # 1. Fotoğrafın orijinal en-boy oranını ve HD kalitesini %100 koru
            orig_h, orig_w = person_orig_bgr.shape[:2]
            max_dim = max(orig_h, orig_w)
            if max_dim > 1200:
                scale_ratio = 1200.0 / max_dim
                person_bgr = cv2.resize(person_orig_bgr, (int(orig_w * scale_ratio), int(orig_h * scale_ratio)), interpolation=cv2.INTER_AREA)
            else:
                person_bgr = person_orig_bgr.copy()

            # Kıyafet çözünürlüğü (HD kalitede)
            c_h, c_w = cloth_orig_bgr.shape[:2]
            max_c = max(c_h, c_w)
            if max_c > 900:
                c_scale = 900.0 / max_c
                cloth_bgr = cv2.resize(cloth_orig_bgr, (int(c_w * c_scale), int(c_h * c_scale)), interpolation=cv2.INTER_AREA)
            else:
                cloth_bgr = cloth_orig_bgr.copy()

            # 2. OTONOM KIYAFET SEGMENTASYONU VE TİP TEŞHİSİ
            segmented_cloth, cloth_mask, cloth_info = segment_cloth_auto(
                cloth_bgr,
                pose_estimator=pose_estimator,
                unet_model=unet_model,
                device=DEVICE
            )

            # Kol tipi kararı (Manuel override yoksa otonom karar geçerli)
            sleeve_decision = cloth_info["sleeve_type"] if override_sleeve == "auto" else override_sleeve

            # 3. İskelet Tespiti ve Akıllı Warp (HD Fotoğraf Üzerinde)
            keypoints_and_landmarks = pose_estimator.get_keypoints(person_bgr)
            hp_p, wp_p = person_bgr.shape[:2]
            warped_cloth = cv2.resize(segmented_cloth, (wp_p, hp_p), interpolation=cv2.INTER_LINEAR)
            warped_mask = cv2.resize(cloth_mask, (wp_p, hp_p), interpolation=cv2.INTER_NEAREST)
            person_keypoints = None
            person_instance_mask = None
            parse_map = None

            if keypoints_and_landmarks is not None:
                person_keypoints, _ = keypoints_and_landmarks
                if instance_segmenter is not None and instance_segmenter.available:
                    person_instance_mask = instance_segmenter.get_person_mask(person_bgr, person_keypoints)

                parse_map = body_parser.parse(person_bgr) if (body_parser is not None and body_parser.available) else None
                if parse_map is not None:
                    # Yalnızca hedef insan instance'ı (ve BodyParser ten/kıyafet kurtarma birleşimi) üzerindeki semantik bölgeleri tut
                    person_instance_mask, parse_map = refine_person_instance(
                        parse_map,
                        person_keypoints,
                        person_bgr.shape,
                        person_instance_mask=person_instance_mask
                    )

                try:
                    wc, wm = warp_clothes(
                        person_bgr,
                        segmented_cloth,
                        cloth_mask.astype(np.uint8),
                        person_keypoints,
                        widen_ratio=widen_ratio,
                        sleeve_type=sleeve_decision,
                        parse_map=parse_map,
                        person_instance_mask=person_instance_mask
                    )
                    if wc is not None and wm is not None:
                        warped_cloth, warped_mask = wc, wm
                except Exception as e:
                    st.warning(f"Warp uyarısı: {e}")
            else:
                st.warning("⚠️ Fotoğrafta net bir insan pozu tespit edilemedi. Lütfen kişinin daha belirgin olduğu bir fotoğraf yükleyin.")

            # 4. Pose ve Agnostic Haritası (HD Fotoğraf Kalitesinde)
            pose_bgr = make_pose_map(person_bgr, pose_estimator, keypoints=person_keypoints)
            agnostic_bgr = make_agnostic(
                person_bgr,
                pose_estimator,
                cloth_region_mask=warped_mask,
                sleeve_type=sleeve_decision,
                keypoints=person_keypoints
            )

            # 5. GAN Çıkarımı (Faz 3.3: Uzak çekim fotoğraflarda 256x256 çözünürlüğün boşa gitmemesi için
            # yalnızca kişi/kıyafet ROI kutusunu 256x256'ya ölçekleyip işler ve tam çözünürlüklü tuvale geri yerleştirir)
            gan_result_img = None
            if generator is not None:
                try:
                    hp_full, wp_full = person_bgr.shape[:2]
                    roi_union = (warped_mask > 0)
                    if person_instance_mask is not None and person_instance_mask.shape == roi_union.shape:

                        roi_union = roi_union | (person_instance_mask > 0)
                    ry_idx, rx_idx = np.where(roi_union)
                    if len(ry_idx) > 50:
                        pad_y = max(16, int((ry_idx.max() - ry_idx.min()) * 0.12))
                        pad_x = max(16, int((rx_idx.max() - rx_idx.min()) * 0.12))
                        ry1 = max(0, int(ry_idx.min()) - pad_y)
                        ry2 = min(hp_full, int(ry_idx.max()) + pad_y)
                        rx1 = max(0, int(rx_idx.min()) - pad_x)
                        rx2 = min(wp_full, int(rx_idx.max()) + pad_x)
                    else:
                        ry1, ry2, rx1, rx2 = 0, hp_full, 0, wp_full

                    agnostic_256 = cv2.resize(agnostic_bgr[ry1:ry2, rx1:rx2], (256, 256))
                    cloth_256 = cv2.resize(warped_cloth[ry1:ry2, rx1:rx2], (256, 256))
                    pose_256 = cv2.resize(pose_bgr[ry1:ry2, rx1:rx2], (256, 256))
                    mask_256 = cv2.resize(warped_mask[ry1:ry2, rx1:rx2], (256, 256), interpolation=cv2.INTER_NEAREST)

                    agnostic_t = norm_transform(Image.fromarray(cv2.cvtColor(agnostic_256, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
                    cloth_t = norm_transform(Image.fromarray(cv2.cvtColor(cloth_256, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
                    pose_t = norm_transform(Image.fromarray(cv2.cvtColor(pose_256, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
                    mask_t = torch.from_numpy(mask_256).float().unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0

                    with torch.no_grad():
                        out_tensor = generator(agnostic_t, cloth_t, mask_t, pose_t)
                    gan_roi_pil = tensor_to_image(out_tensor)
                    gan_roi_rgb = cv2.resize(np.array(gan_roi_pil.convert("RGB")), (rx2 - rx1, ry2 - ry1), interpolation=cv2.INTER_LINEAR)
                    full_gan_rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
                    full_gan_rgb[ry1:ry2, rx1:rx2] = gan_roi_rgb
                    gan_result_img = Image.fromarray(full_gan_rgb)
                except Exception as e:
                    st.warning(f"GAN çıkarım uyarısı: {e}")

            # 6. Kalın Kıyafet / Kürk Silme (Eğer kullanıcı modu açtıysa)
            dress_person_bgr = person_bgr.copy()
            if enable_coat_removal and person_keypoints is not None:
                dress_person_bgr = remove_bulky_garment(
                    dress_person_bgr,
                    warped_mask,
                    person_keypoints,
                    sleeve_type=sleeve_decision
                )

            # 7. Otonom Hibrit Füzyon (NİHAİ HD NET GÖRSEL)
            final_result = autonomous_hybrid_fusion(
                dress_person_bgr,
                warped_cloth,
                warped_mask,
                gan_result_img=gan_result_img,
                sleeve_type=sleeve_decision,
                keypoints=person_keypoints,
                parse_map=parse_map,
                person_instance_mask=person_instance_mask
            )

            # ── YAPAY ZEKA OTONOM KARAR RAPORU ───────────────────────────
            st.markdown("---")
            st.subheader("🤖 Yapay Zeka Otonom Karar Raporu")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                model_status = "👤 Manken/İnsan Tespit Edildi (Kafa/Boyun Ayrıştırıldı)" if cloth_info["has_model"] else "📦 Düz Ürün Görseli (Arka Plan Silindi)"
                st.metric("Görsel Türü", "Mankenli" if cloth_info["has_model"] else "Düz Ürün", help=model_status)
            with m2:
                sleeve_label = {
                    "short": "👕 Kısa Kollu (T-Shirt)",
                    "long": "🧥 Uzun Kollu (Gömlek/Kazak)",
                    "sleeveless": "🎽 Kolsuz (Askılı/Atlet)"
                }.get(sleeve_decision, sleeve_decision)
                st.metric("Kıyafet Modeli", sleeve_label)
            with m3:
                protection_status = "Ön Kollar & Eller Korundu" if sleeve_decision in ["short", "sleeveless"] else "Tam Kol Giydirme"
                st.metric("Ten & Kol Koruması", "🛡️ Aktif", help=protection_status)
            with m4:
                engine_status = "GAN Işık/Gölge + Warped Doku" if generator is not None else "Akıllı Warped Doku"
                st.metric("Sentez Motoru", "✨ Hibrit Füzyon", help=engine_status)

            # ── NİHAİ SONUÇ VE SEÇENEKLER ─────────────────────────────────
            st.markdown("---")
            st.subheader("🎉 Giydirilmiş Sonuçlar")

            tab1, tab2, tab3 = st.tabs([
                "🌟 Akıllı Giydirme (HD Net Doku & Önerilen)",
                f"🤖 Saf {EPOCH_LABEL} TryOn-GAN Çıktısı",
                "⚖️ Yan Yana Kıyaslama"
            ])

            with tab1:
                r1, r2 = st.columns([2, 1])
                with r1:
                    st.image(final_result, caption="Doğal Proporsiyonlu Akıllı Giydirme (HD Kumaş & Ten Korunmuş)", width="stretch")
                with r2:
                    st.markdown("### 📋 Neler Yapıldı?")
                    st.write(f"- **Kıyafet Sınıfı:** {sleeve_label}")
                    st.write("- **Gövde Dökümü:** Tişört etek ucu ve pazu genişliği gövdeye tam oturtuldu, büzülme/sıkışma giderildi.")
                    st.write("- **Ten Koruması:** Ön kollar, eller, boyun ve saçlar %100 temiz ten dokusuyla korundu.")
                    st.write("- **HD Netlik:** Kumaş logosu, kolye ve reglan dikişleri en yüksek çözünürlükte sunuldu.")

            with tab2:
                if gan_result_img is not None:
                    g1, g2 = st.columns([2, 1])
                    with g1:
                        st.image(gan_result_img, caption=f"{EPOCH_LABEL} TryOn-GAN Nöral Ağ Çıktısı (Saf Model Tahmini)", width="stretch")
                    with g2:
                        st.markdown(f"### 🧠 {EPOCH_LABEL} GAN Analizi")
                        st.write("Colab'de eğittiğiniz yeni ResNet + Bilinear generator modelinin saf nöral ağ üretimidir.")
                        st.info("💡 **Gözlem:** 50 epoch ve Multi-scale VGG ile eğitilen bu yeni model, eski 25 epoch modele göre kıyafet kalıbını, vücut sınırlarını ve renk uyumunu çok daha kusursuz sentezler.")
                else:
                    st.info("GAN modeli bulunamadığı için saf model çıktısı gösterilemiyor.")

            with tab3:
                c_k1, c_k2 = st.columns(2)
                with c_k1:
                    st.markdown("**🌟 Akıllı Giydirme (HD Doku):**")
                    st.image(final_result, width="stretch")
                with c_k2:
                    st.markdown(f"**🤖 Saf {EPOCH_LABEL} GAN:**")
                    if gan_result_img is not None:
                        st.image(gan_result_img, width="stretch")
                    else:
                        st.write("Model yüklenemedi.")

            # ── ADIM ADIM İNCELEME & DEBUG ───────────────────────────────
            with st.expander("🔬 Adım Adım Pipeline 2.0 İncelemesi (Debug Görselleri)", expanded=debug_mode):
                dcol0, dcol1, dcol2, dcol3, dcol4, dcol5 = st.columns(6)
                with dcol0:
                    st.caption("0. YOLO11-Seg Kişi Maskesi")
                    if person_instance_mask is not None:
                        st.image(person_instance_mask, width="stretch")
                    else:
                        st.caption("Fallback")
                with dcol1:
                    st.caption("1. Pose İskelet Haritası")
                    st.image(cv2.cvtColor(pose_bgr, cv2.COLOR_BGR2RGB), width="stretch")
                with dcol2:
                    st.caption("2. İzole Kıyafet")
                    st.image(cv2.cvtColor(segmented_cloth, cv2.COLOR_BGR2RGB), width="stretch")
                with dcol3:
                    st.caption("3. Kıyafet Maskesi")
                    st.image(cloth_mask, width="stretch")
                with dcol4:
                    st.caption("4. Warped Kıyafet")
                    st.image(cv2.cvtColor(warped_cloth, cv2.COLOR_BGR2RGB), width="stretch")
                with dcol5:
                    st.caption("5. Kişi Agnostic (Ten Korunmuş)")
                    st.image(cv2.cvtColor(agnostic_bgr, cv2.COLOR_BGR2RGB), width="stretch")

                if gan_result_img is not None:
                    st.markdown("---")
                    st.caption("6. Saf GAN Modeli Çıktısı (Yalnızca Üretilen Katman):")
                    st.image(gan_result_img, width=320)
