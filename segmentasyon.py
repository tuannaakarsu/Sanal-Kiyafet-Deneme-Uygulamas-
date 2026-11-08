"""
segmentasyon.py
Kıyafet ve insan segmentasyonu için çoklu yöntem desteği:
1. rembg (u2net_cloth_seg - üst kıyafet, alt kıyafet, elbise ayrıştırma)
2. rembg (u2netp - genel arka plan silme)
3. MediaPipe Pose destekli kafa/boyun filtreleme (manken üzerindeki kıyafetler için)
4. Klasik UNet (3, 59)
5. Renk / Arka Plan bazlı segmentasyon
"""

import cv2
import numpy as np
from PIL import Image
import torch

try:
    import rembg
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

_CLOTH_SESSION = None
_GENERAL_SESSION = None


def get_cloth_session():
    global _CLOTH_SESSION
    if _CLOTH_SESSION is None and REMBG_AVAILABLE:
        try:
            _CLOTH_SESSION = rembg.new_session("u2net_cloth_seg")
        except Exception:
            _CLOTH_SESSION = None
    return _CLOTH_SESSION


def get_general_session():
    global _GENERAL_SESSION
    if _GENERAL_SESSION is None and REMBG_AVAILABLE:
        try:
            _GENERAL_SESSION = rembg.new_session("u2netp")
        except Exception:
            _GENERAL_SESSION = None
    return _GENERAL_SESSION


def clean_binary_mask(mask: np.ndarray, min_area_ratio: float = 0.05) -> np.ndarray:
    """Morfolojik operasyonlar ve en büyük bağlı bileşeni tutma (yaka oyuntusunu koruyan filtreleme)."""
    m = (mask > 127).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = 1 + int(np.argmax(areas))
        m = np.where(labels == largest_idx, 255, 0).astype(np.uint8)

    # Tüm kapalı iç delikleri (göğüs baskısı / logo içindeki beyaz alanlar) doldur;
    # yaka açıklığı üstten dış arka plana açık olduğu için (h_info[3] == -1) doğal olarak korunur:
    contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        for i, h_info in enumerate(hierarchy[0]):
            if h_info[3] >= 0:  # Kapalı iç delik
                cv2.drawContours(m, [contours[i]], -1, 255, thickness=cv2.FILLED)
    return m


def remove_head_and_neck(mask: np.ndarray, cloth_bgr: np.ndarray, pose_estimator=None) -> np.ndarray:
    """
    Eğer kıyafet görselinde bir manken/insan varsa;
    kafa, çene, boyun teni ve kolye bölgesini maskeden temizler.
    Böylece mankenin yüzü veya boyun derisi hedef kişinin gövdesine yapışmaz!
    """
    h, w = mask.shape
    cleaned_mask = mask.copy()

    # 1. BodyParser ile mankenin saç (1), yüz (3), ten (2) ve alt kıyafet/pantolon (5) piksellerini
    # kontur boyunca hassas biçimde maskeden çıkar (yatay giyotin kesimi yapmadan yaka ve omuz eğimini korur):
    p_map = None
    try:
        from vucut_analizi import BodyParser
        bp = BodyParser()
        if bp.available:
            p_map = bp.parse(cloth_bgr)
            if p_map is not None:
                # Saç (1), ten (2), yüz (3) ve alt kıyafet (5) çıkarılır; üst kıyafet (4) korunur
                human_non_upper = np.isin(p_map, [1, 2, 3, 5])
                if np.any(human_non_upper):
                    cleaned_mask[human_non_upper] = 0
    except Exception:
        pass

    # 2. MediaPipe Pose ile kontrol et (kafa dairesi ve yaka içi boyun teni temizliği)
    if pose_estimator is not None and getattr(pose_estimator, "available", False):
        res = pose_estimator.get_keypoints(cloth_bgr)
        if res is not None:
            keypoints, _ = res
            # 11: Sol omuz, 12: Sağ omuz
            if len(keypoints) > 12:
                ls_x, ls_y, ls_v = keypoints[11][0], keypoints[11][1], keypoints[11][2]
                rs_x, rs_y, rs_v = keypoints[12][0], keypoints[12][1], keypoints[12][2]
                if ls_v > 0.35 and rs_v > 0.35:
                    shoulder_y = min(ls_y, rs_y)
                    sh_w = max(20.0, float(abs(rs_x - ls_x)))
                    neck_cx = float((ls_x + rs_x) / 2.0)

                    # Yalnızca burun gerçekten tespit edilmişse ve omuzlardan anlamlı biçimde yukarıdaysa kafa dairesini temizle
                    if len(keypoints) > 0 and keypoints[0][2] > 0.45 and keypoints[0][1] < (shoulder_y - sh_w * 0.25):
                        nose_x, nose_y = int(keypoints[0][0]), int(keypoints[0][1])
                        head_r = max(10, int(abs(shoulder_y - nose_y) * 0.55))
                        cv2.circle(cleaned_mask, (nose_x, nose_y), head_r, 0, -1)
                        neck_cx = 0.6 * neck_cx + 0.4 * float(nose_x)

                    # Mankenin boyun derisi / kolyesi yaka V-açıklığında kalmışsa temizle:
                    y_neck_top = 0
                    y_neck_bot = min(h, int(max(ls_y, rs_y) + sh_w * 0.12))
                    x_neck_l = max(0, int(neck_cx - sh_w * 0.20))
                    x_neck_r = min(w, int(neck_cx + sh_w * 0.20))
                    if y_neck_bot > y_neck_top and x_neck_r > x_neck_l:
                        roi_bgr = cloth_bgr[y_neck_top:y_neck_bot, x_neck_l:x_neck_r]
                        roi_ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
                        skin_in_collar = (
                            (roi_ycrcb[:, :, 0] > 60) &
                            (roi_ycrcb[:, :, 1] >= 136) & (roi_ycrcb[:, :, 1] <= 176) &
                            (roi_ycrcb[:, :, 2] >= 77) & (roi_ycrcb[:, :, 2] <= 127)
                        )
                        rh_n, rw_n = roi_bgr.shape[:2]
                        yg, xg = np.mgrid[0:rh_n, 0:rw_n]
                        collar_wedge = (np.abs(xg - (rw_n / 2.0)) <= (rw_n * 0.46 * (1.0 - 0.45 * (yg / max(1.0, float(rh_n))))))
                        rem_neck = skin_in_collar & collar_wedge
                        if np.any(rem_neck):
                            rem_u8 = cv2.dilate(rem_neck.astype(np.uint8) * 255, np.ones((3, 3), np.uint8), iterations=1)
                            sub_m = cleaned_mask[y_neck_top:y_neck_bot, x_neck_l:x_neck_r]
                            sub_m[rem_u8 > 0] = 0
                            cleaned_mask[y_neck_top:y_neck_bot, x_neck_l:x_neck_r] = sub_m

    return clean_binary_mask(cleaned_mask)


def _has_human_in_cloth(cloth_bgr: np.ndarray, pose_estimator=None) -> bool:
    """Kıyafet görselinde gerçek bir insan/manken olup olmadığını MediaPipe Pose ve BodyParser ten analizi ile kontrol eder."""
    h, w = cloth_bgr.shape[:2]
    pe = pose_estimator
    if pe is None:
        try:
            from posetespiti import PoseEstimator
            pe = PoseEstimator()
        except Exception:
            pe = None
    if pe is not None and getattr(pe, "available", False):
        try:
            res = pe.get_keypoints(cloth_bgr)
            if res is not None:
                kps, _ = res
                if len(kps) > 16:
                    ls_v, rs_v = float(kps[11][2]), float(kps[12][2])
                    nose_v = float(kps[0][2]) if len(kps) > 0 else 0.0
                    wr_v = max(float(kps[15][2]), float(kps[16][2]))
                    if nose_v > 0.45 or (ls_v > 0.35 and rs_v > 0.35 and wr_v > 0.30):
                        return True
        except Exception:
            pass

    corner_pts = np.array([
        cloth_bgr[0, 0], cloth_bgr[0, w - 1],
        cloth_bgr[h - 1, 0], cloth_bgr[h - 1, w - 1],
        cloth_bgr[0, w // 2]
    ], dtype=np.float32)
    # Düz beyaz/açık veya tamamen siyah arka planlı (ve Pose'da kafa/el bulunmayan) katalog ürünlerinde manken yoktur
    if (corner_pts.mean() > 175.0 and corner_pts.std(axis=0).max() <= 22.0) or corner_pts.max() <= 6.0:
        return False

    # Kafası kırpılmış mankenli fotoğraflarda (ör. c_raglan) MediaPipe Pose kafa olmadığı için None dönebilir.
    # Gerçek ortam arka planlı görselde BodyParser hem boyun/kol teni (2) hem kıyafet (4) buluyorsa insan vardır:
    try:
        from vucut_analizi import BodyParser
        bp = BodyParser()
        if bp.available:
            p_map = bp.parse(cloth_bgr)
            if p_map is not None:
                skin_ratio = float(np.mean(p_map == 2))
                cloth_ratio = float(np.mean(p_map == 4))
                if skin_ratio > 0.025 and cloth_ratio > 0.12:
                    return True
    except Exception:
        pass
    return False


def _enforce_flatlay_symmetry(mask: np.ndarray, cloth_bgr: np.ndarray, has_model: bool = False):
    """
    Mankensiz (flat-lay / katalog) kıyafetlerde sol-sağ simetri onarımı:
    Çizgili veya gölgeli ürünlerde bir tarafın göğüs/kol bölgesi arka plan sanılarak ısırılmışsa
    (ör. mavi-beyaz çizgili uzun kollu kazakta sol göğüs/kol kaybı ve merkez ekseni aşan ısırık),
    sağlam tarafın konturunu ve yatay satır dokusunu yansıtarak maskeyi ve eksik kumaş dokusunu tamamlar.
    Sağlam olan ürünlerde (ör. kırmızı-siyah çizgili yazılı kazak) göğüs yazılarının ters dönmemesi için
    yalnızca ciddi asimetrik oyulma (< 0.56 genişlik oranı, >= %16 satır) tespit edildiğinde devreye girer.
    """
    if has_model or mask is None or cloth_bgr is None:
        return cloth_bgr, mask

    ys, xs = np.where(mask > 0)
    if len(xs) < 200:
        return cloth_bgr, mask

    h, w = mask.shape[:2]
    y0, y1 = int(ys.min()), int(ys.max())
    bh = max(1, y1 - y0)

    # Üst yaka/omuz bandından (%2-%15) güvenilir simetri ekseni (cx) hesapla
    top_ys, top_xs = np.where(mask[int(y0 + bh * 0.02):int(y0 + bh * 0.15), :] > 0)
    if len(top_xs) > 20:
        cx = 0.5 * (float(np.percentile(top_xs, 3)) + float(np.percentile(top_xs, 97)))
    else:
        cx = 0.5 * float(xs.min() + xs.max())

    cxi = int(round(cx))
    if cxi < int(w * 0.22) or cxi > int(w * 0.78):
        return cloth_bgr, mask

    # Yalnızca gerçek ve derin bir asimetrik oyulma/ısırık varsa (%12 - %78 dikey bantta < 0.56 oran) devreye gir:
    r_y1, r_y2 = int(y0 + bh * 0.12), int(y0 + bh * 0.78)
    asymmetric_rows = 0
    total_checked = max(1, r_y2 - r_y1)
    for y in range(r_y1, r_y2):
        row_xs = np.where(mask[y, :] > 0)[0]
        if len(row_xs) < 8:
            continue
        w_l = max(0.0, cx - float(row_xs.min()))
        w_r = max(0.0, float(row_xs.max()) - cx)
        if max(w_l, w_r) > 12.0 and min(w_l, w_r) < 0.56 * max(w_l, w_r):
            asymmetric_rows += 1

    if asymmetric_rows < max(6, int(total_checked * 0.16)):
        return cloth_bgr, mask

    x_grid = np.arange(w, dtype=np.int32)
    mirror_x = np.clip(2 * cxi - x_grid, 0, w - 1)
    mirrored_mask = mask[:, mirror_x].copy()

    repaired_mask = cv2.bitwise_or(mask, mirrored_mask)
    y_cuff = int(y0 + bh * 0.72)
    if y_cuff < h:
        repaired_mask[y_cuff:, :] = cv2.morphologyEx(
            repaired_mask[y_cuff:, :], cv2.MORPH_CLOSE, np.ones((27, 17), np.uint8), iterations=1
        )
    repaired_mask = clean_binary_mask(repaired_mask)

    # Sağlam yarıdaki iç kumaş parlaklığını ölç:
    # Eğer kıyafette siyah/koyu çizgiler varsa (fg_p15 < 75.0), siyah çizgileri korumak için min_valid_luma = 10.0 tut;
    # yalnızca açık renkli (ör. mavi-beyaz) ürünlerde ısırık kenarındaki koyu haleyi dışlamak için yüksek eşik kullan:
    right_intact = np.sum(mask[:, cxi:] > 0) >= np.sum(mask[:, :cxi] > 0)
    intact_half = np.zeros_like(mask, dtype=bool)
    if right_intact:
        intact_half[int(y0 + bh * 0.20):int(y0 + bh * 0.80), cxi:] = True
    else:
        intact_half[int(y0 + bh * 0.20):int(y0 + bh * 0.80), :cxi] = True
    eroded_base = cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=1) > 0
    intact_sample = eroded_base & intact_half
    if np.sum(intact_sample) > 50:
        fg_p15 = float(np.percentile(cloth_bgr[intact_sample].max(axis=1), 15))
        min_valid_luma = max(45.0, fg_p15 * 0.88) if fg_p15 >= 75.0 else 10.0
    else:
        min_valid_luma = 12.0

    raw_fg = ((mask > 0) & (cloth_bgr.max(axis=2) >= min_valid_luma)).astype(np.uint8) * 255
    eroded_fg = cv2.erode(raw_fg, np.ones((11, 11), np.uint8), iterations=1)
    clean_fg = eroded_fg > 0

    repaired_bgr = cloth_bgr.copy()
    mirrored_bgr = cloth_bgr[:, mirror_x]
    mirrored_clean = clean_fg[:, mirror_x]

    need_tex = (repaired_mask > 0) & (~clean_fg)
    from_mirror = need_tex & mirrored_clean
    if np.any(from_mirror):
        repaired_bgr[from_mirror] = mirrored_bgr[from_mirror]

    still_missing = need_tex & (~mirrored_clean)
    if np.any(still_missing):
        filled_now = clean_fg | from_mirror
        for y in range(0, h):
            miss_xs = np.where(still_missing[y, :])[0]
            if len(miss_xs) == 0:
                continue
            valid_xs = np.where(clean_fg[y, :])[0]
            if len(valid_xs) < 6:
                valid_xs = np.where(filled_now[y, :])[0]
            if len(valid_xs) < 6:
                for dy in range(1, 30):
                    for yn in (y - dy, y + dy):
                        if 0 <= yn < h:
                            v_near = np.where(clean_fg[yn, :])[0]
                            if len(v_near) >= 6:
                                valid_xs = v_near
                                src_strip = repaired_bgr[yn, valid_xs]
                                break
                    if len(valid_xs) >= 6:
                        break
                else:
                    src_strip = None
            else:
                src_strip = repaired_bgr[y, valid_xs]

            if src_strip is not None and len(src_strip) >= 4:
                n_src = len(src_strip)
                rel_idx = np.abs(miss_xs - int(valid_xs[0])) % max(1, (2 * n_src - 2))
                refl_idx = np.where(rel_idx >= n_src, (2 * n_src - 2) - rel_idx, rel_idx)
                refl_idx = np.clip(refl_idx, 0, n_src - 1)
                repaired_bgr[y, miss_xs] = src_strip[refl_idx]

    return repaired_bgr, repaired_mask


def segment_cloth_rembg(cloth_bgr: np.ndarray, cloth_type="upper", pose_estimator=None):
    """
    rembg u2net_cloth_seg ile kıyafeti arka plandan ve mankenden ayıklar.
    cloth_type: 'upper' (üst kıyafet), 'lower' (alt kıyafet), 'full' (elbise), 'auto'
    Döndürür: (isolated_cloth_bgr, binary_mask)
    """
    h, w = cloth_bgr.shape[:2]
    corners = np.array([
        cloth_bgr[0, 0], cloth_bgr[0, w - 1],
        cloth_bgr[h - 1, 0], cloth_bgr[h - 1, w - 1]
    ], dtype=np.float32)
    black_ratio = float(np.mean(cloth_bgr.max(axis=2) <= 6))
    if corners.max() <= 6.0 and 0.10 < black_ratio < 0.88:
        has_human = _has_human_in_cloth(cloth_bgr, pose_estimator=pose_estimator)
        mask = (cloth_bgr.max(axis=2) > 8).astype(np.uint8) * 255
        mask = clean_binary_mask(mask)
        mask = _remove_corner_remnants(mask, cloth_bgr)
        if not has_human:
            cloth_bgr, mask = _enforce_flatlay_symmetry(mask, cloth_bgr, has_model=False)
        isolated_cloth = cv2.bitwise_and(cloth_bgr, cloth_bgr, mask=mask)
        return isolated_cloth, mask

    cloth_rgb = cv2.cvtColor(cloth_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cloth_rgb)

    session = get_cloth_session()
    mask = None

    if session is not None:
        try:
            # u2net_cloth_seg çıktısı 3 katmanlıdır: [Upper, Lower, Full]
            out_img = rembg.remove(pil_img, session=session)
            out_arr = np.array(out_img)

            if out_arr.shape[0] == 3 * h:
                # 0: Upper, 1: Lower, 2: Full
                if cloth_type == "lower":
                    slice_img = out_arr[h:2*h, :, :]
                elif cloth_type == "full":
                    slice_img = out_arr[2*h:3*h, :, :]
                else: # upper or auto
                    slice_img = out_arr[:h, :, :]

                alpha = slice_img[:, :, 3]
                if np.sum(alpha > 15) > (h * w * 0.02):
                    mask = (alpha > 15).astype(np.uint8) * 255
        except Exception as e:
            print(f"Cloth seg hatasi: {e}")

    # Eğer u2net_cloth_seg yeterli maske bulamadıysa (örneğin düz kıyafet / masa üstü ürün):
    # u2netp (genel arka plan silici) kullan
    if mask is None or np.sum(mask > 0) < (h * w * 0.03):
        gen_session = get_general_session()
        if gen_session is not None:
            try:
                out_img = rembg.remove(pil_img, session=gen_session)
                out_arr = np.array(out_img)
                alpha = out_arr[:, :, 3]
                mask = (alpha > 15).astype(np.uint8) * 255
            except Exception as e:
                print(f"General rembg hatasi: {e}")

    has_human = _has_human_in_cloth(cloth_bgr, pose_estimator=pose_estimator)

    if mask is None:
        # YALNIZCA görselde gerçekten bir insan/manken varsa VE düz stüdyo arka planlı değilse BodyParser kullan!
        # Mankensiz çizgili kazak/tişörtlerde BodyParser (Selfie Multiclass) çalıştırmak kıyafetin yarısını siler.
        corner_pts = np.array([
            cloth_bgr[0, 0], cloth_bgr[0, w - 1],
            cloth_bgr[h - 1, 0], cloth_bgr[h - 1, w - 1],
            cloth_bgr[0, w // 2]
        ], dtype=np.float32)
        is_uniform_studio_bg = (corner_pts.std(axis=0).max() <= 14.0)
        if has_human and not is_uniform_studio_bg:
            try:
                from vucut_analizi import BodyParser
                bp = BodyParser()
                if bp.available:
                    p_map = bp.parse(cloth_bgr)
                    if p_map is not None:
                        bp_mask = np.isin(p_map, [4, 5]).astype(np.uint8) * 255
                        if np.sum(bp_mask > 0) > (h * w * 0.04):
                            mask = bp_mask
            except Exception:
                pass
        if mask is None:
            # Mankensiz / düz ürün fotoğraflarında kenar-tohumlu renk ve çizgi korumalı segmentasyon kullan
            _, mask = segment_cloth_color(cloth_bgr)

    # İnsan/manken varsa kafa/boyun kısmını temizle, yoksa flat-lay simetri onarımı uygula
    if has_human:
        mask = remove_head_and_neck(mask, cloth_bgr, pose_estimator=pose_estimator)
    mask = clean_binary_mask(mask)
    mask = _remove_corner_remnants(mask, cloth_bgr)
    if not has_human:
        cloth_bgr, mask = _enforce_flatlay_symmetry(mask, cloth_bgr, has_model=False)

    isolated_cloth = cv2.bitwise_and(cloth_bgr, cloth_bgr, mask=mask)
    return isolated_cloth, mask


def _remove_corner_remnants(mask: np.ndarray, cloth_bgr: np.ndarray) -> np.ndarray:
    """Koyu renkli kıyafetlerin alt bölgesinde ve dış köşelerinde kalan parlak arka plan / eşofman / atlet artıklarını temizler."""
    if mask is None or cloth_bgr is None:
        return mask
    h, w = mask.shape[:2]
    corner_pts = np.array([
        cloth_bgr[0, 0], cloth_bgr[0, w - 1],
        cloth_bgr[h - 1, 0], cloth_bgr[h - 1, w - 1],
        cloth_bgr[0, w // 2]
    ], dtype=np.float32)
    # Düz beyaz/açık stüdyo arka planlı katalog ürünlerinde (urun2, urun4, urun5) eşofman/atlet artığı yoktur;
    # göğüs/etek baskılarını (Blondie vb.) korumak için doğrudan döndür:
    if corner_pts.mean() > 180.0 and corner_pts.std(axis=0).max() <= 14.0:
        return mask

    ys_m, xs_m = np.where(mask > 0)
    if len(xs_m) > 100:
        y0_m, y1_m = int(ys_m.min()), int(ys_m.max())
        x0_m, x1_m = int(xs_m.min()), int(xs_m.max())
        bh_m = max(1, y1_m - y0_m)
        bw_m = max(1, x1_m - x0_m)
        y_s1, y_s2 = int(y0_m + bh_m * 0.35), int(y0_m + bh_m * 0.72)
        x_s1, x_s2 = int(x0_m + bw_m * 0.28), int(x0_m + bw_m * 0.72)
        mid_region = mask[y_s1:y_s2, x_s1:x_s2] > 0
        if np.any(mid_region):
            torso_gray = cv2.cvtColor(cloth_bgr[y_s1:y_s2, x_s1:x_s2], cv2.COLOR_BGR2GRAY)
            median_luma = float(np.median(torso_gray[mid_region]))
            if median_luma < 125.0:
                full_gray = cv2.cvtColor(cloth_bgr, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(cloth_bgr, cv2.COLOR_BGR2HSV)
                bot_zone = np.zeros_like(mask, dtype=bool)
                bot_zone[int(y0_m + bh_m * 0.48):, :] = True
                outer_or_hem = (
                    (np.arange(w)[None, :] < (x0_m + bw_m * 0.24)) |
                    (np.arange(w)[None, :] > (x0_m + bw_m * 0.70)) |
                    (np.arange(h)[:, None] > (y0_m + bh_m * 0.76))
                )
                white_remnant = (mask > 0) & bot_zone & outer_or_hem & (full_gray > 112) & (hsv[:, :, 1] < 55)
                if np.any(white_remnant):
                    rem_u8 = cv2.dilate(white_remnant.astype(np.uint8) * 255, np.ones((5, 5), np.uint8), iterations=1)
                    mask = mask.copy()
                    mask[rem_u8 > 0] = 0
                y_hem_open = int(y0_m + bh_m * 0.82)
                if y_hem_open < h - 5:
                    mask[y_hem_open:, :] = cv2.morphologyEx(mask[y_hem_open:, :], cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
                mask = clean_binary_mask(mask)
    return mask


def segment_cloth_unet(cloth_bgr: np.ndarray, unet_model, device="cpu", pose_estimator=None):
    """
    Eğitilmiş 59 sınıflı UNet modeli ile maske tahmini.
    """
    if unet_model is None:
        return segment_cloth_rembg(cloth_bgr, pose_estimator=pose_estimator)

    h, w = cloth_bgr.shape[:2]
    cloth_resized = cv2.resize(cloth_bgr, (256, 256))
    cloth_rgb = cv2.cvtColor(cloth_resized, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(cloth_rgb).permute(2, 0, 1).float() / 255.0
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        output = unet_model(x)
        pred = torch.argmax(output[0], dim=0).cpu().numpy().astype(np.uint8)

    unique, counts = np.unique(pred, return_counts=True)
    freq = {int(k): int(v) for k, v in zip(unique, counts) if int(k) != 0}

    if freq:
        dominant = max(freq, key=freq.get)
        mask = np.where(pred == dominant, 255, 0).astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = clean_binary_mask(mask)
        mask = remove_head_and_neck(mask, cloth_bgr, pose_estimator=pose_estimator)
        isolated = cv2.bitwise_and(cloth_bgr, cloth_bgr, mask=mask)
        return isolated, mask

    return segment_cloth_color(cloth_bgr)


def segment_cloth_color(cloth_bgr: np.ndarray):
    """
    Düz arka planlı (beyaz, gri vb.) ve çizgili/desenli ürünler için
    kenar-bağlantılı stüdyo gölgesi filtreli ve yatay çizgi köprülemeli eşikleme.
    """
    h, w = cloth_bgr.shape[:2]
    corners = np.array([
        cloth_bgr[0, 0], cloth_bgr[0, w-1],
        cloth_bgr[h-1, 0], cloth_bgr[h-1, w-1],
        cloth_bgr[0, w//2]
    ], dtype=np.float32)
    bg_color = corners.mean(axis=0)
    bg_luma = float(bg_color.mean())

    diff = np.abs(cloth_bgr.astype(np.float32) - bg_color).sum(axis=2)
    mask = (diff > 45.0).astype(np.uint8) * 255

    # Açık renk stüdyo arka planlarında (bg_luma > 180) kol altı ve etek altındaki açık gri stüdyo gölgelerini
    # (parlaklığı > 165 ve doygunluğu düşük nötr gri gölge) maskeden çıkar:
    if bg_luma > 180.0:
        gray = cv2.cvtColor(cloth_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(cloth_bgr, cv2.COLOR_BGR2HSV)
        studio_shadow = (gray > 165) & (hsv[:, :, 1] < 28) & (diff < 180.0)
        mask[studio_shadow] = 0

    # Yatay çizgili kıyafetlerde (ör. mavi-beyaz veya kırmızı-beyaz çizgili kazak) beyaz çizgiler
    # stüdyo arka planıyla yakın renkte olduğu için maskede yatay kopukluklar oluşturabilir.
    # Merkez sütunlardaki dikey 0<->255 geçiş sayısını ölçerek çizgili kumaşı tespit et:
    mid_cols = (mask[:, int(w * 0.30):int(w * 0.70)] > 0).astype(np.int8)
    col_transitions = np.sum(np.abs(np.diff(mid_cols, axis=0)), axis=0)
    if len(col_transitions) > 0 and np.percentile(col_transitions, 75) >= 6:
        # Çizgili kıyafet: koyu çizgiler arasındaki açık/beyaz şeritleri dikey morfolojik kapanışla birleştir
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((41, 7), np.uint8), iterations=1)

    mask = clean_binary_mask(mask)
    isolated = cv2.bitwise_and(cloth_bgr, cloth_bgr, mask=mask)
    return isolated, mask


def detect_cloth_attributes(cloth_bgr: np.ndarray, cloth_mask: np.ndarray, pose_estimator=None) -> dict:
    """
    Kıyafet görseli ve maskesini inceleyerek:
    1. Görselde manken/insan var mı?
    2. Kol tipi nedir (kısa kol, uzun kol, kolsuz)?
    kararlarını %100 otonom olarak üretir.
    """
    h, w = cloth_mask.shape[:2]
    decisions = {
        "has_model": False,
        "sleeve_type": "short",
        "label_tr": "Kısa Kollu (T-Shirt)",
        "confidence": 0.92,
        "details": {}
    }

    # 1. Manken / İnsan Algılama (MediaPipe Pose)
    if pose_estimator is not None and getattr(pose_estimator, "available", False):
        res = pose_estimator.get_keypoints(cloth_bgr)
        if res is not None:
            keypoints, _ = res
            if len(keypoints) > 12:
                ls_v = keypoints[11][2]
                rs_v = keypoints[12][2]
                nose_v = keypoints[0][2] if len(keypoints) > 0 else 0
                if (ls_v > 0.22 and rs_v > 0.22) or nose_v > 0.3:
                    decisions["has_model"] = True
                    decisions["details"]["model_detected"] = True

                    # Manken üzerinden kol analizi
                    ls_x = keypoints[11][0]
                    rs_x = keypoints[12][0]
                    torso_left = min(ls_x, rs_x) - 15
                    torso_right = max(ls_x, rs_x) + 15

                    # Gerçek uzun kol tespiti: Kol kumaşı dirsekten bileğe kadar uzanmalı VE gövdenin dışında olmalı
                    long_sleeve_evidence = 0
                    for el_idx, wr_idx in [(13, 15), (14, 16)]:
                        if len(keypoints) > wr_idx:
                            el = keypoints[el_idx]
                            wr = keypoints[wr_idx]
                            if el[2] > 0.25 and wr[2] > 0.25:
                                is_on_torso = (torso_left < wr[0] < torso_right)
                                mid_arm = (el[:2] + wr[:2]) / 2.0
                                mx, my = int(mid_arm[0]), int(mid_arm[1])
                                wx, wy = int(wr[0]), int(wr[1])

                                if 0 <= mx < w and 0 <= my < h and 0 <= wx < w and 0 <= wy < h:
                                    if not is_on_torso and cloth_mask[wy, wx] > 100 and cloth_mask[my, mx] > 100:
                                        long_sleeve_evidence += 1

                    if long_sleeve_evidence >= 1:
                        decisions["sleeve_type"] = "long"
                        decisions["label_tr"] = "Uzun Kollu (Gömlek / Kazak)"
                        return decisions

    if not decisions["has_model"] and _has_human_in_cloth(cloth_bgr, pose_estimator=pose_estimator):
        decisions["has_model"] = True
        decisions["details"]["model_detected"] = True

    # 2. Geometrik / Maske Profil Analizi (Düz ürün veya genel analiz)
    ys, xs = np.where(cloth_mask > 0)
    if len(xs) > 100:
        ymin, ymax = int(ys.min()), int(ys.max())
        xmin, xmax = int(xs.min()), int(xs.max())
        bbox_h = max(ymax - ymin, 1)
        bbox_w = max(xmax - xmin, 1)

        aspect_ratio = bbox_w / max(bbox_h, 1)

        # Üst omuz/kol bölgesi (%18 - %36)
        # Orta gövde / koltuk altı bölgesi (%45 - %58)
        # Alt gövde / etek bölgesi (%68 - %86)
        upper_slice = cloth_mask[int(ymin + bbox_h * 0.18):int(ymin + bbox_h * 0.36), :]
        mid_slice = cloth_mask[int(ymin + bbox_h * 0.45):int(ymin + bbox_h * 0.58), :]
        bot_slice = cloth_mask[int(ymin + bbox_h * 0.68):int(ymin + bbox_h * 0.86), :]

        def get_span(sl):
            cols = np.where(sl > 0)[1]
            return (cols.max() - cols.min()) if len(cols) > 0 else 0

        w_upper = get_span(upper_slice)
        w_mid = get_span(mid_slice)
        w_bot = get_span(bot_slice)
        mid_ratio = float(w_mid) / max(float(w_upper), 1.0)
        bot_ratio = float(w_bot) / max(float(w_upper), 1.0)

        # Dış kol sütunlarının (%8 sol ve %8 sağ en dış kenar) dikey uzanımı (outer_reach):
        # Kısa kollu tişörtlerde (ör. urun2, urun4, urun5) dış kol uçları üst yarıda (%20-%48) biter ve
        # orta/alt gövde genişliği üst omuz/kol genişliğine göre belirgin daralır (mid_ratio < 0.83, bot_ratio < 0.83).
        # Uzun kollu gömlek/kazaklarda ise kollar alt yarıya kadar indiği için ya kol-gövde ayrımı (cleft) oluşur
        # ya da alt yarı genişliği üst yarı kadar geniş kalır (mid_ratio >= 0.84, bot_ratio >= 0.84, outer_reach > 0.60).
        outer_w = max(4, int(bbox_w * 0.08))
        left_col_ys = np.where(cloth_mask[ymin:ymax + 1, xmin:xmin + outer_w] > 0)[0]
        right_col_ys = np.where(cloth_mask[ymin:ymax + 1, max(xmin, xmax - outer_w):xmax + 1] > 0)[0]
        reach_l = float(left_col_ys.max()) / float(bbox_h) if len(left_col_ys) > 8 else 0.0
        reach_r = float(right_col_ys.max()) / float(bbox_h) if len(right_col_ys) > 8 else 0.0
        outer_reach = max(reach_l, reach_r)

        # Koltuk altı ayrımı (cleft): %42-%80 dikey bandında kol ile gövde arasında boşluk olan satır sayısı
        cleft_rows = 0
        for y_r in range(int(ymin + bbox_h * 0.42), int(ymin + bbox_h * 0.80)):
            row_m = (cloth_mask[y_r, xmin:xmax + 1] > 0).astype(np.int8)
            transitions = np.sum(np.abs(np.diff(row_m)))
            if transitions >= 4:  # En az 2 ayrı parça (kol + boşluk + gövde)
                cleft_rows += 1
        has_sleeve_cleft = cleft_rows >= max(5, int(bbox_h * 0.05))

        decisions["details"]["aspect_ratio"] = round(aspect_ratio, 3)
        decisions["details"]["mid_ratio"] = round(mid_ratio, 3)
        decisions["details"]["bot_ratio"] = round(bot_ratio, 3)
        decisions["details"]["outer_reach"] = round(outer_reach, 3)
        decisions["details"]["cleft_rows"] = int(cleft_rows)

        # Tişört ve Kol tipi analizi:
        # 1. En/Boy oranı aşırı dar ve üst omuz kol çıkıntısı gövdeden darsa kolsuz atlet
        if aspect_ratio < 0.54 or (aspect_ratio < 0.65 and w_upper < w_mid * 0.82):
            decisions["sleeve_type"] = "sleeveless"
            decisions["label_tr"] = "Kolsuz (Atlet / Askılı)"
        # 2. Uzun kollu tespiti (mankensiz/düz ürünlerde dış kol uzanımı + alt genişlik korunumu veya koltuk altı yarığı)
        elif not decisions["has_model"] and (
            (has_sleeve_cleft and outer_reach > 0.55) or
            (outer_reach > 0.60 and (mid_ratio > 0.84 or bot_ratio > 0.84) and aspect_ratio > 0.70) or
            (bot_ratio > 0.86 and mid_ratio > 0.86 and aspect_ratio > 0.72) or
            (aspect_ratio > 1.30 and bot_ratio > 0.90)
        ):
            decisions["sleeve_type"] = "long"
            decisions["label_tr"] = "Uzun Kollu (Gömlek / Kazak / Ceket)"
        else:
            # Standart üst giyim varsayılanı tişört / polo
            decisions["sleeve_type"] = "short"
            decisions["label_tr"] = "Kısa Kollu (T-Shirt / Polo)"

    return decisions


def segment_cloth_auto(cloth_bgr: np.ndarray, pose_estimator=None, unet_model=None, device="cpu"):
    """
    Kullanıcıya hiçbir ayar sormadan:
    1. Manken var mı tespit eder (varsa yüzü/boynu temizler).
    2. En uygun segmentasyon yöntemini sırayla dener.
    3. Kol tipini (kısa kol / uzun kol / kolsuz) otonom belirler.
    Döndürür: (isolated_cloth, cloth_mask, cloth_info)
    """
    h, w = cloth_bgr.shape[:2]
    total_px = h * w

    # 1. rembg cloth-seg dene
    isolated, mask = segment_cloth_rembg(cloth_bgr, cloth_type="upper", pose_estimator=pose_estimator)
    method_used = "rembg_cloth"

    mask_area = np.sum(mask > 0)
    if mask_area < (total_px * 0.04):
        isolated, mask = segment_cloth_rembg(cloth_bgr, cloth_type="auto", pose_estimator=None)
        method_used = "rembg_general"
        mask_area = np.sum(mask > 0)

    if mask_area < (total_px * 0.04) and unet_model is not None:
        isolated, mask = segment_cloth_unet(cloth_bgr, unet_model, device=device, pose_estimator=pose_estimator)
        method_used = "unet"
        mask_area = np.sum(mask > 0)

    if mask_area < (total_px * 0.04):
        isolated, mask = segment_cloth_color(cloth_bgr)
        method_used = "color"

    # 2. Otonom Özellik ve Kol Analizi
    cloth_info = detect_cloth_attributes(cloth_bgr, mask, pose_estimator=pose_estimator)
    cloth_info["method_used"] = method_used

    # 3. Manken tespit edildiyse kafa/boyun temizlenir; mankensiz ise simetri onarımı uygulanır
    base_bgr = isolated if (isolated is not None and np.any(isolated > 0)) else cloth_bgr
    if cloth_info["has_model"]:
        mask = remove_head_and_neck(mask, base_bgr, pose_estimator=pose_estimator)
    else:
        base_bgr, mask = _enforce_flatlay_symmetry(mask, base_bgr, has_model=False)

    mask = _remove_corner_remnants(mask, base_bgr)
    isolated = cv2.bitwise_and(base_bgr, base_bgr, mask=mask)

    return isolated, mask, cloth_info


# ─────────────────────────────────────────────────────────────────────────────
# GERİYE DÖNÜK UYUMLULUK EXPORTLARI (anakod.py / app.py için)
# ─────────────────────────────────────────────────────────────────────────────
def load_model(weight_path, device="cpu"):
    """Geriye dönük uyumluluk için UNet ağırlık yükleyici."""
    try:
        try:
            from modelunet import UNet
        except ImportError:
            from model import UNet
        model = UNet(3, 59)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"UNet yükleme uyarısı: {e}")
        return None


def segment_cloth(image, model=None, device="cpu"):
    """Geriye dönük uyumluluk için kıyafet segmentasyonu wrapper'ı."""
    try:
        isolated, mask, _ = segment_cloth_auto(image, unet_model=model, device=device)
        return mask
    except Exception as e:
        print(f"Segmentasyon hatası, renk filtresine düşülüyor: {e}")
        _, mask = segment_cloth_color(image)
        return mask


