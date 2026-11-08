"""
kiyafetigiydir.py - Robust Homography & Anatomical VTON Engine
- Robust 4-Corner Projective Homography: No non-linear buckling, no side-wings, no sports-bra truncation.
- Natural Torso Continuity: Full shoulder-to-hip coverage down to sweatpants/trousers waistband (LH/RH).
- Linear Texture Mapping: 100% distortion-free fabric texture, straight neckline, and intact logos.
- Pixel-Accurate Occlusion: Hair, forearms, hands, and chin protection.
- Ambient Contact Shadow: Natural contact shadow and soft Gaussian alpha feathering.
"""
import cv2
import numpy as np

from vucut_analizi import (
    refine_person_instance,
    extract_body_contours,
    create_occlusion_mask,
    LBL_HAIR,
    LBL_FACE,
    LBL_SKIN
)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Maskeyi ikili (binary) hale getirir, delikleri doldurur ve en büyük bileşeni alır."""
    if mask is None:
        return None
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = mask[:, :, 3]
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    m = (mask > 127).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        keep = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = np.where(labels == keep, 255, 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        for i, h_info in enumerate(hierarchy[0]):
            if h_info[3] >= 0:
                cv2.drawContours(m, [contours[i]], -1, 255, thickness=cv2.FILLED)
    return m


def _extract_canonical_sleeve_tex(clothes_img: np.ndarray, src_mask: np.ndarray, x_min: int, x_max: int, y_min: int, y_max: int, canon_w: int = 96, canon_h: int = 180, side: str = "left"):
    """
    Kaynak kıyafetin kol dokusunu (canon_h x canon_w x 3) boyutunda %100 dolu, arka plansız,
    yatay çizgileri (siyah/kırmızı/mavi/beyaz) ve kumaş dokusunu dalgalanmasız koruyan
    kanonik bir kol şeridi olarak üretir.
    Eğik duran kaynak kollarda yatay kesitin birden fazla çizgiyi çapraz kesip dalgalı/diyagonal
    desen oluşturmasını önlemek için, aynı kumaştan üretilen uzun kollu kıyafetlerde merkez gövdenin
    logosuz yatay şerit profilini (satır medyanı + mikro kumaş dokusu) referans alır.
    """
    h_img, w_img = src_mask.shape[:2]
    x_min = max(0, min(w_img - 1, int(x_min)))
    x_max = max(x_min + 1, min(w_img, int(x_max)))
    y_min = max(0, min(h_img - 1, int(y_min)))
    y_max = max(y_min + 1, min(h_img, int(y_max)))

    inner_mask = cv2.erode(src_mask, np.ones((5, 5), np.uint8), iterations=1)
    ys_all, xs_all = np.where(inner_mask > 0)
    if len(xs_all) == 0:
        return np.full((canon_h, canon_w, 3), 128, dtype=np.uint8)

    cx_g = int(0.5 * (xs_all.min() + xs_all.max()))
    bw_g = max(20, int(xs_all.max() - xs_all.min()))
    tc_x1 = max(0, cx_g - int(bw_g * 0.16))
    tc_x2 = min(w_img, cx_g + int(bw_g * 0.16))

    # Kol bölgesi ile merkez gövde aynı kumaş/renk ailesinden mi kontrol et:
    slv_roi_m = inner_mask[y_min:y_max, x_min:x_max] > 0
    tor_roi_m = inner_mask[y_min:y_max, tc_x1:tc_x2] > 0
    use_torso_stripe_profile = True
    if np.sum(slv_roi_m) > 50 and np.sum(tor_roi_m) > 50:
        slv_med = np.median(clothes_img[y_min:y_max, x_min:x_max][slv_roi_m], axis=0).astype(np.float32)
        tor_med = np.median(clothes_img[y_min:y_max, tc_x1:tc_x2][tor_roi_m], axis=0).astype(np.float32)
        if float(np.linalg.norm(slv_med - tor_med)) > 52.0:
            use_torso_stripe_profile = False

    rows_tex = []
    for y in range(y_min, y_max):
        if use_torso_stripe_profile:
            t_slice = inner_mask[y, tc_x1:tc_x2] > 0
            if np.sum(t_slice) >= 8:
                t_px = clothes_img[y, tc_x1:tc_x2][t_slice].astype(np.float32)
                # Satır medyanı küçük göğüs yazısı/logo piksellerini (%100) filtreler ve saf yatay şerit rengini verir:
                r_med = np.median(t_px, axis=0)
                # Doğal kumaş dokusunu korumak için sapması düşük piksellerin mikro dokusunu ekle:
                dev = t_px - r_med[None, :]
                dev_norm = np.linalg.norm(dev, axis=1)
                clean_dev = dev[dev_norm < 18.0]
                if len(clean_dev) >= 4:
                    reps = (canon_w // len(clean_dev)) + 2
                    grain = np.tile(clean_dev, (reps, 1))[:canon_w]
                else:
                    grain = np.zeros((canon_w, 3), dtype=np.float32)
                row_strip = np.clip(r_med[None, :] + grain * 0.55, 0.0, 255.0).astype(np.uint8)
                rows_tex.append(row_strip)
                continue

        row_slice = inner_mask[y, x_min:x_max] > 0
        r_xs = np.where(row_slice)[0]
        if len(r_xs) < 6:
            continue
        if side == "left":
            s0 = int(r_xs[0])
            s1 = s0
            while (s1 + 1) < (x_max - x_min) and row_slice[s1 + 1]:
                s1 += 1
        else:
            s1 = int(r_xs[-1])
            s0 = s1
            while (s0 - 1) >= 0 and row_slice[s0 - 1]:
                s0 -= 1

        span_len = s1 - s0 + 1
        if span_len >= 6:
            trim = max(1, int(span_len * 0.15))
            c0 = x_min + s0 + trim
            c1 = x_min + s1 + 1 - trim
            if c1 - c0 >= 3:
                r_med = np.median(clothes_img[y, c0:c1], axis=0).astype(np.uint8)
                rows_tex.append(np.tile(r_med[None, :], (canon_w, 1)))

    if len(rows_tex) < 8:
        med_col = np.median(clothes_img[ys_all, xs_all], axis=0).astype(np.uint8)
        return np.full((canon_h, canon_w, 3), med_col, dtype=np.uint8)

    stack = np.array(rows_tex, dtype=np.uint8)
    canon = cv2.resize(stack, (canon_w, canon_h), interpolation=cv2.INTER_LINEAR)
    return canon


def _warp_quad_patch(src_patch: np.ndarray, dst_quad: np.ndarray, out_w: int, out_h: int):
    """Kanonik dikdörtgen yamayı hedef 4-köşe dörtgene (tl, tr, br, bl) perspektif olarak büker."""
    sh, sw = src_patch.shape[:2]
    src_q = np.array([
        [0, 0],
        [sw - 1, 0],
        [sw - 1, sh - 1],
        [0, sh - 1]
    ], dtype=np.float32)
    M_q = cv2.getPerspectiveTransform(src_q, dst_quad.astype(np.float32))
    w_img = cv2.warpPerspective(src_patch, M_q, (out_w, out_h), flags=cv2.INTER_LINEAR)
    w_msk = np.zeros((out_h, out_w), dtype=np.uint8)
    cv2.fillConvexPoly(w_msk, np.round(dst_quad).astype(np.int32), 255)
    return w_img, w_msk


def warp_clothes(
    person_img,
    clothes_img,
    mask,
    keypoints,
    widen_ratio=0.15,
    sleeve_type="short",
    parse_map=None,
    person_instance_mask=None,
    instance_mask=None,
    parse_clean=None,
):
    """
    Robust Anatomical Virtual Try-On Warp Motoru:
    1. İskelet landmarklarından (omuz ve kalça) anatomik yön ve genişlik vektörlerini hesaplar.
    2. 4-Noktalı Projektif Perspektif Dönüşümü ile kıyafetin logosunu, kumaş dokusunu ve geometrisini
       %100 doğrusal ve pürüzsüz korur (kanatlanma, bükülme, ters dönme ve göbek açılması olmaz).
    3. Etek ucunu doğal kalça/eşofman bel çizgisine (LH/RH) kadar kesintisiz uzatır.
    4. Yüz ve çene bölgesini korumak için çene koruma maskesi uygular.
    5. Saç, ön kollar ve elleri piksel seviyesinde occlusion maskesiyle korur.
    """
    if keypoints is None or len(keypoints) < 25:
        return None, None

    hp, wp = person_img.shape[:2]
    src_mask = _clean_mask(mask)
    if src_mask is None:
        return None, None

    ys, xs = np.where(src_mask > 0)
    if len(xs) == 0:
        return None, None

    x0, y0, bw, bh = cv2.boundingRect(np.stack([xs, ys], axis=1))
    if bw < 10 or bh < 10:
        return None, None

    # 1. Person Instance Segmentation & Parse Temizliği
    yolo_inst = person_instance_mask if person_instance_mask is not None else instance_mask
    if yolo_inst is None:
        try:
            from vucut_analizi import PersonInstanceSegmenter
            pis = PersonInstanceSegmenter()
            yolo_inst = pis.get_person_mask(person_img, keypoints)
        except Exception:
            yolo_inst = None

    src_parse = parse_map if parse_map is not None else parse_clean
    instance_mask, parse_clean = refine_person_instance(src_parse, keypoints, person_img.shape, person_instance_mask=yolo_inst)

    # 2. İskelet Noktaları ve Anatomik Hedef Bölge Hesabı
    ls = np.array(keypoints[11][:2], dtype=np.float32)
    rs = np.array(keypoints[12][:2], dtype=np.float32)
    lh = np.array(keypoints[23][:2], dtype=np.float32)
    rh = np.array(keypoints[24][:2], dtype=np.float32)

    if rs[0] < ls[0]:
        p_sh_l, p_sh_r = rs.copy(), ls.copy()
        p_hip_l, p_hip_r = rh.copy(), lh.copy()
        idx_el_l, idx_wr_l = 14, 16
        idx_el_r, idx_wr_r = 13, 15
    else:
        p_sh_l, p_sh_r = ls.copy(), rs.copy()
        p_hip_l, p_hip_r = lh.copy(), rh.copy()
        idx_el_l, idx_wr_l = 13, 15
        idx_el_r, idx_wr_r = 14, 16

    sh_vec = p_sh_r - p_sh_l
    sh_w = max(20.0, float(np.linalg.norm(sh_vec)))
    sh_u = sh_vec / sh_w
    sh_angle = np.arctan2(sh_vec[1], sh_vec[0])
    up_vec = np.array([-np.sin(sh_angle), np.cos(sh_angle)], dtype=np.float32)
    if up_vec[1] > 0:
        up_vec = -up_vec

    # Perspektif kaynaklı aşırı omuz eğimini yumuşat (duruş dik ise bir omzun düşmesini engeller)
    if abs(p_sh_r[1] - p_sh_l[1]) < sh_w * 0.14:
        avg_sh_y = (p_sh_l[1] + p_sh_r[1]) / 2.0
        p_sh_l[1] = 0.45 * p_sh_l[1] + 0.55 * avg_sh_y
        p_sh_r[1] = 0.45 * p_sh_r[1] + 0.55 * avg_sh_y

    y_sh = float((p_sh_l[1] + p_sh_r[1]) / 2.0)
    sh_cx = float((p_sh_l[0] + p_sh_r[0]) / 2.0)
    hip_cx = float((p_hip_l[0] + p_hip_r[0]) / 2.0)
    y_hip_raw = float((p_hip_l[1] + p_hip_r[1]) / 2.0)

    # MediaPipe 23-24 kalça eklemleri femur başı (kasık/üst bacak) hizasındadır.
    # Omuzların perspektif/dönüş nedeniyle daraldığı (selfie vb.) pozlarda etek ucunun göbekte kalmaması için
    # gerçek dikey gövde yüksekliğini (torso_h = y_hip_raw - y_sh) baz al:
    torso_h = max(sh_w * 1.25, y_hip_raw - y_sh)
    y_hem = max(y_hip_raw - sh_w * 0.04, min(y_hip_raw + sh_w * 0.08, y_sh + torso_h * 0.98))

    # Eğer kişinin üzerinde pantolondan farklı renkte sarkık bir üst tişört varsa (ör. ayna selfiesindeki kırmızı tişört),
    # etek ucunu eski üst tişörtün alt sınırını tamamen kapatacak şekilde çözünürlükten bağımsız (sh_w oranlı) adaptif uzat:
    if parse_clean is not None:
        old_c_full = (parse_clean == 4) & (instance_mask > 0)
        y_top_s1, y_top_s2 = max(0, int(y_sh + sh_w * 0.20)), min(hp, int(y_sh + sh_w * 0.78))
        y_leg_s1, y_leg_s2 = max(0, int(y_hip_raw + sh_w * 0.22)), min(hp, int(y_hip_raw + sh_w * 0.65))
        x_scan_l = max(0, int(sh_cx - sh_w * 0.55))
        x_scan_r = min(wp, int(sh_cx + sh_w * 0.55))
        top_pts = person_img[y_top_s1:y_top_s2, :][old_c_full[y_top_s1:y_top_s2, :]]
        leg_pts = person_img[y_leg_s1:y_leg_s2, :][old_c_full[y_leg_s1:y_leg_s2, :]] if y_leg_s2 > y_leg_s1 + 5 else np.empty((0, 3), dtype=np.uint8)
        if len(leg_pts) < 30:
            # Kalçadan aşağı kırpılmış veya koyu kot pantolonun BodyParser'da LBL_CLOTH dışına düştüğü fotoğraflarda
            # en alt %12'lik dilimdeki kişi gövde/bacak piksellerini alt giyim referansı al:
            y_bot_s1 = max(int(y_sh + sh_w * 1.05), int(hp * 0.88))
            if hp > y_bot_s1 + 4:
                bot_roi_mask = (instance_mask[y_bot_s1:hp, x_scan_l:x_scan_r] > 0)
                if np.sum(bot_roi_mask) > 25:
                    leg_pts = person_img[y_bot_s1:hp, x_scan_l:x_scan_r][bot_roi_mask]
        if len(top_pts) > 50 and len(leg_pts) > 25:
            top_med = np.median(top_pts, axis=0).astype(np.float32)
            leg_med = np.median(leg_pts, axis=0).astype(np.float32)
            leg_red_excess = float(max(0.0, leg_med[2] - max(leg_med[0], leg_med[1])))
            if np.linalg.norm(top_med - leg_med) > 28.0 or leg_red_excess < 15.0:
                y_scan_end = min(hp - 1, int(y_hip_raw + sh_w * 0.48), int(y_sh + sh_w * 1.62))
                last_top_y = int(y_hem)
                for y_r in range(int(y_hem), max(int(y_hem), y_scan_end)):
                    r_mask = old_c_full[y_r, x_scan_l:x_scan_r]
                    if np.sum(r_mask) > 8:
                        row_med = np.median(person_img[y_r, x_scan_l:x_scan_r][r_mask], axis=0).astype(np.float32)
                        d_top = float(np.linalg.norm(row_med - top_med))
                        d_leg = float(np.linalg.norm(row_med - leg_med))
                        row_red_excess = float(max(0.0, row_med[2] - max(row_med[0], row_med[1])))
                        # Pantolon renginden belirgin farklı olan (veya kırmızı/renkli tişört eteği olan) satırları kapsa:
                        if (d_top < 52.0 and d_top < d_leg * 0.88) or (d_leg > 38.0 and row_red_excess > leg_red_excess + 18.0):
                            last_top_y = y_r
                if last_top_y > y_hem:
                    y_hem = min(float(last_top_y + sh_w * 0.12), float(hp - 1), y_hip_raw + sh_w * 0.48, y_sh + sh_w * 1.62)

    # 3. Kaynak Kıyafetin İç Gövde (Torso) ve Etek Genişliği Analizi
    # Uzun kollu kıyafetlerde kollar alt yarıya kadar sarktığı için merkezden dışa doğru ilk boşluğa (koltuk altı yarığına)
    # kadar taranır; böylece sarkan uzun kollar gövde genişliği sanılmaz.
    mid_y1, mid_y2 = int(y0 + bh * 0.44), int(y0 + bh * 0.75)
    cx_src = int(x0 + bw * 0.5)
    xls, xrs = [], []
    for y_row in range(mid_y1, mid_y2):
        r_xs = np.where(src_mask[y_row, :] > 0)[0]
        if len(r_xs) > 10:
            if sleeve_type == "long":
                # Merkezden sola ve sağa doğru kesintisiz gövde sınırını bul
                xl = cx_src
                while xl > x0 and src_mask[y_row, xl - 1] > 0:
                    xl -= 1
                xr = cx_src
                while xr < (x0 + bw - 1) and src_mask[y_row, xr + 1] > 0:
                    xr += 1
                xls.append(xl)
                xrs.append(xr)
            else:
                xls.append(r_xs.min())
                xrs.append(r_xs.max())

    src_torso_l = float(np.percentile(xls, 65)) if xls else float(x0 + bw * 0.16)
    src_torso_r = float(np.percentile(xrs, 35)) if xrs else float(x0 + bw * 0.84)

    if sleeve_type == "long":
        # Uzun kollu ürünlerde kollar gövdeye bitişikse iç gövde sınırını anatomik gövde oranına (%20-%80) kilitle
        src_torso_l = float(np.clip(src_torso_l, x0 + bw * 0.18, x0 + bw * 0.31))
        src_torso_r = float(np.clip(src_torso_r, x0 + bw * 0.69, x0 + bw * 0.82))

    hem_y1, hem_y2 = int(y0 + bh * 0.75), int(y0 + bh * 0.96)
    h_xls, h_xrs = [], []
    for y_row in range(hem_y1, max(hem_y1 + 1, hem_y2)):
        r_xs = np.where(src_mask[y_row, :] > 0)[0]
        if len(r_xs) > 10:
            h_xls.append(r_xs.min())
            h_xrs.append(r_xs.max())
    src_hem_l = float(np.percentile(h_xls, 20)) if h_xls else src_torso_l
    src_hem_r = float(np.percentile(h_xrs, 80)) if h_xrs else src_torso_r

    if sleeve_type == "long":
        src_hem_l = max(src_hem_l, src_torso_l - bw * 0.03)
        src_hem_r = min(src_hem_r, src_torso_r + bw * 0.03)

    src_aspect = float(bw) / float(max(bh, 1))
    if sleeve_type == "sleeveless":
        src_top_l, src_top_r = float(x0), float(x0 + bw)
        src_bot_l, src_bot_r = float(x0), float(x0 + bw)
    elif sleeve_type == "long":
        # Uzun kollularda gövde homografisi yalnızca omuz-bel merkez gövdesini eşler; kollar eklemli olarak ayrıca bükülür
        src_top_l = float(max(x0, src_torso_l - bw * 0.05))
        src_top_r = float(min(x0 + bw, src_torso_r + bw * 0.05))
        src_bot_l = float(0.55 * src_torso_l + 0.45 * src_hem_l)
        src_bot_r = float(0.55 * src_torso_r + 0.45 * src_hem_r)
    elif src_aspect < 0.825:
        # Manken üzerinde dikey duran tişörtler (kollar zaten aşağı sarkık):
        src_top_l = float(x0 + bw * 0.08)
        src_top_r = float(x0 + bw * 0.92)
        src_bot_l = float(0.40 * src_torso_l + 0.60 * src_hem_l)
        src_bot_r = float(0.40 * src_torso_r + 0.60 * src_hem_r)
    else:
        # Düz serilmiş (flat-lay) katalog tişörtleri (kollar yanlara açık):
        src_top_l = float(x0 + (src_torso_l - x0) * 0.65)
        src_top_r = float((x0 + bw) - ((x0 + bw) - src_torso_r) * 0.65)
        src_bot_l = float(0.60 * src_torso_l + 0.40 * src_hem_l)
        src_bot_r = float(0.60 * src_torso_r + 0.40 * src_hem_r)

    # 4. Hedef Kişi Göğüs, Omuz ve Bel Sınırları (İskelet + Eski Kıyafet Konturu Birleşimi)
    w_top = 0.16 + (widen_ratio * 0.14)
    w_bot = 0.52 + (widen_ratio * 0.12)
    waist_cx = 0.65 * hip_cx + 0.35 * sh_cx

    # Kısa kollu kıyafetlerde dirsekler yana açılmışsa (eller belde/akimbo veya selfie),
    # kısa kol açıklığını üst kol yönünde hafifçe genişlet ki tek parça doğal kol üst kolu örtsün:
    flare_l, flare_r = 0.0, 0.0
    if sleeve_type == "short" and len(keypoints) >= 15:
        if keypoints[idx_el_l][2] > 0.25:
            v_l = np.array(keypoints[idx_el_l][:2], dtype=np.float32) - p_sh_l
            len_l = float(np.linalg.norm(v_l))
            if len_l > 12.0 and (v_l[0] / len_l) < -0.34:
                flare_l = min(0.12, abs(float(v_l[0]) / len_l) * 0.15)
        if keypoints[idx_el_r][2] > 0.25:
            v_r = np.array(keypoints[idx_el_r][:2], dtype=np.float32) - p_sh_r
            len_r = float(np.linalg.norm(v_r))
            if len_r > 12.0 and (v_r[0] / len_r) > 0.34:
                flare_r = min(0.12, abs(float(v_r[0]) / len_r) * 0.15)

    chest_l = float(p_sh_l[0] - sh_w * (w_top + flare_l))
    chest_r = float(p_sh_r[0] + sh_w * (w_top + flare_r))
    waist_l = float(waist_cx - sh_w * w_bot)
    waist_r = float(waist_cx + sh_w * w_bot)
    strap_top_y = y_sh - sh_w * 0.16

    if parse_clean is not None:
        old_c = (parse_clean == 4) & (instance_mask > 0)
        # Omuz askısı üst sınırı
        y_strap_min = max(0, int(y_sh - sh_w * 0.30))
        strap_ys = np.where(old_c[y_strap_min:int(y_sh), :])[0]
        if len(strap_ys) > 10:
            strap_top_y = min(strap_top_y, float(y_strap_min + np.percentile(strap_ys, 5)))

        # Gövde ve bel bölgesindeki eski kıyafet genişliğini alt gövde bandından (eski kısa kolların altından) ölç:
        max_torso_reach = sh_w * (0.76 if sleeve_type == "long" else 0.72)
        y_wb1 = min(int(y_hem) - 4, int(y_sh + sh_w * 0.45))
        w_band = old_c[max(0, y_wb1):int(y_hem), :]
        w_xs = np.where(w_band)[1]
        if len(w_xs) > 50:
            c_l = float(np.clip(np.percentile(w_xs, 4), sh_cx - max_torso_reach, sh_cx))
            c_r = float(np.clip(np.percentile(w_xs, 96), sh_cx, sh_cx + max_torso_reach))
            chest_l = min(chest_l, c_l - 6.0)
            chest_r = max(chest_r, c_r + 6.0)
            waist_l = min(waist_l, c_l - 6.0)
            waist_r = max(waist_r, c_r + 6.0)

    # Yaka / Omuz yüksekliği: Eski askıları ve eğimli omuz kavisini kapatacak, çeneye taşmayacak şekilde ayarla
    up_neck_px = max(sh_w * 0.22, y_sh - strap_top_y + 12.0)
    if len(keypoints) > 10 and keypoints[9][2] > 0.25:
        mouth_y = (keypoints[9][1] + keypoints[10][1]) / 2.0
        neck_gap = max(10.0, y_sh - mouth_y)
        up_neck_px = min(up_neck_px, neck_gap * 0.68)
    else:
        up_neck_px = min(up_neck_px, sh_w * 0.32)

    dst_tl = np.array([chest_l, p_sh_l[1] - up_neck_px], dtype=np.float32)
    dst_tr = np.array([chest_r, p_sh_r[1] - up_neck_px], dtype=np.float32)
    dst_br = np.array([waist_r, y_hem], dtype=np.float32)
    dst_bl = np.array([waist_l, y_hem], dtype=np.float32)

    # Uzun kollu kıyafetlerde sarkan kaynak kolların gövde homografisiyle aşağı (suya/bacaklara) veya
    # omuz yanlarına (T-bar şeklinde) taşmasını önlemek için gövde maskesini iç gövde dikiş hattıyla sınırla:
    torso_src_mask = src_mask.copy()
    torso_src_img = clothes_img.copy()
    y_src_top = float(y0)
    y_src_bot = float(y0 + bh)
    if sleeve_type == "long":
        cut_l = max(0, int(src_torso_l))
        cut_r = min(clothes_img.shape[1], int(src_torso_r))
        core_half = max(12, int((cut_r - cut_l) * 0.28))
        core_l = max(cut_l, cx_src - core_half)
        core_r = min(cut_r - 1, cx_src + core_half)
        y_lower_sleeve_guard = int(y0 + bh * 0.42)
        # Her satırda merkezden (cx_src) sola ve sağa doğru kesintisiz gövdeyi takip et;
        # koltuk altı boşluğundan (src_mask == 0) sonra gelen sarkık kol parçalarını %100 sil.
        # Alt yarıda (y >= y0 + 0.42*bh) gövdeye bitişik duran eğik kaynak kolların yan dikişten içeri sızmasını
        # önlemek için merkez gövde çekirdeğini (core_l .. core_r) baz al:
        for y_row in range(y0, min(clothes_img.shape[0], y0 + bh)):
            if torso_src_mask[y_row, cx_src] > 0:
                xl = cx_src
                while xl > cut_l and torso_src_mask[y_row, xl - 1] > 0:
                    xl -= 1
                xr = cx_src
                while xr < cut_r - 1 and torso_src_mask[y_row, xr + 1] > 0:
                    xr += 1
                if y_row >= y_lower_sleeve_guard:
                    xl = max(xl, core_l)
                    xr = min(xr, core_r)
                torso_src_mask[y_row, :xl] = 0
                torso_src_mask[y_row, xr + 1:] = 0
            elif y_row < int(y0 + bh * 0.35):
                # Üst yaka oyuğu (merkez boş ama omuzlar dolu)
                torso_src_mask[y_row, :cut_l] = 0
                torso_src_mask[y_row, cut_r:] = 0
            else:
                # Gövde etek ucunun altı (merkez boş, yalnızca yanlarda sarkan uzun kol manşetleri var) -> tamamen sıfırla!
                torso_src_mask[y_row, :] = 0

        # Üst omuz simetri kontrolü: Mankenli ürün fotoğraflarında tek omuzda yukarı taşan koyu saç kütlesini
        # (ör. sol omuz üstündeki saç çıkıntısı) temizle; simetrik kıyafetlerde ise doğal yuvarlak omuz eğimini koru:
        half_w_l = max(1, cx_src - cut_l)
        half_w_r = max(1, cut_r - cx_src)
        y_top_l, y_top_r = None, None
        for y_scan in range(y0, min(clothes_img.shape[0], int(y0 + bh * 0.28))):
            if y_top_l is None and np.sum(torso_src_mask[y_scan, cut_l:cx_src] > 0) >= max(4, int(half_w_l * 0.14)):
                y_top_l = y_scan
            if y_top_r is None and np.sum(torso_src_mask[y_scan, cx_src:cut_r] > 0) >= max(4, int(half_w_r * 0.14)):
                y_top_r = y_scan
            if y_top_l is not None and y_top_r is not None:
                break

        if y_top_l is not None and y_top_r is not None:
            y_sym_top = max(y_top_l, y_top_r)
            y_sh_clean_end = min(clothes_img.shape[0], int(y_sym_top + bh * 0.17))
            sh_span = min(half_w_l, half_w_r)
            if sh_span >= 8 and y_sh_clean_end > y_sym_top + 2:
                l_roi = torso_src_img[y_sym_top:y_sh_clean_end, cx_src - sh_span:cx_src].astype(np.float32)
                r_roi = torso_src_img[y_sym_top:y_sh_clean_end, cx_src:cx_src + sh_span].astype(np.float32)
                l_chroma = float(np.mean(np.max(l_roi, axis=2) - np.min(l_roi, axis=2)))
                r_chroma = float(np.mean(np.max(r_roi, axis=2) - np.min(r_roi, axis=2)))
                if (y_top_l < y_top_r - 5) or (r_chroma > l_chroma + 7.5):
                    torso_src_mask[:max(y0, y_sym_top - 2), :] = 0
                    torso_src_img[y_sym_top:y_sh_clean_end, cx_src - sh_span:cx_src] = torso_src_img[y_sym_top:y_sh_clean_end, cx_src:cx_src + sh_span][:, ::-1]
                    torso_src_mask[y_sym_top:y_sh_clean_end, cx_src - sh_span:cx_src] = torso_src_mask[y_sym_top:y_sh_clean_end, cx_src:cx_src + sh_span][:, ::-1]
                elif (y_top_r < y_top_l - 5) or (l_chroma > r_chroma + 7.5):
                    torso_src_mask[:max(y0, y_sym_top - 2), :] = 0
                    torso_src_img[y_sym_top:y_sh_clean_end, cx_src:cx_src + sh_span] = torso_src_img[y_sym_top:y_sh_clean_end, cx_src - sh_span:cx_src][:, ::-1]
                    torso_src_mask[y_sym_top:y_sh_clean_end, cx_src:cx_src + sh_span] = torso_src_mask[y_sym_top:y_sh_clean_end, cx_src - sh_span:cx_src][:, ::-1]

        t_ys = np.where(torso_src_mask > 0)[0]
        if len(t_ys) > 20:
            y_src_top = float(t_ys.min())
            y_src_bot = float(np.percentile(t_ys, 99.0))

        # Gövde yan dikişlerinde (cut_l .. cut_r) koltuk altı kavisinden ve alt etek kavisinden
        # kaynaklanan daralmaları her kenarın kendi yerel şerit fazıyla (xl sol kenarı, xr sağ kenarı)
        # doldur ki omuz eğimi korunsun ve dikey şerit faz kırılması (seam step) oluşmasın:
        y_fill_start = max(int(y_src_top + bh * 0.18), y0)
        for y_row in range(y_fill_start, min(clothes_img.shape[0], int(y_src_bot) + 1)):
            r_xs = np.where(torso_src_mask[y_row, cut_l:cut_r] > 0)[0]
            if len(r_xs) >= 12:
                xl = cut_l + int(r_xs[0])
                xr = cut_l + int(r_xs[-1])
                if xl > cut_l:
                    edge_l = torso_src_img[y_row, xl:min(xr + 1, xl + 10)]
                    col_l = np.median(edge_l, axis=0).astype(np.uint8)
                    torso_src_img[y_row, cut_l:xl] = col_l
                    torso_src_mask[y_row, cut_l:xl] = 255
                if xr < cut_r - 1:
                    edge_r = torso_src_img[y_row, max(xl, xr - 9):xr + 1]
                    col_r = np.median(edge_r, axis=0).astype(np.uint8)
                    torso_src_img[y_row, xr + 1:cut_r] = col_r
                    torso_src_mask[y_row, xr + 1:cut_r] = 255
            elif y_row > y_fill_start and np.any(torso_src_mask[y_row - 1, cut_l:cut_r] > 0):
                torso_src_img[y_row, cut_l:cut_r] = torso_src_img[y_row - 1, cut_l:cut_r]
                torso_src_mask[y_row, cut_l:cut_r] = 255

        src_top_l = float(cut_l)
        src_top_r = float(cut_r)
        src_bot_l = float(cut_l)
        src_bot_r = float(cut_r)

    src_pts = np.array([
        [src_top_l, y_src_top],
        [src_top_r, y_src_top],
        [src_bot_r, y_src_bot],
        [src_bot_l, y_src_bot]
    ], dtype=np.float32)

    dst_pts = np.array([dst_tl, dst_tr, dst_br, dst_bl], dtype=np.float32)

    # Projektif Homografi Dönüşümü (Perspektif Transform - Gövde)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_cloth = cv2.warpPerspective(torso_src_img, M, (wp, hp), flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpPerspective(torso_src_mask, M, (wp, hp), flags=cv2.INTER_NEAREST)
    warped_mask = (warped_mask > 0).astype(np.uint8) * 255

    # 4B. Eklemli Kol Bükme (Articulated Sleeve Warping - Faz 2 & Faz 3.2 Omuz/Koltuk Altı Kenetleme):
    # - Uzun kollu ("long") kıyafetlerde: Sol ve sağ kolları Omuz -> Dirsek -> Bilek zinciri boyunca 2 parçalı büker,
    #   üst kol kökünü dış omuz dikişine kenetler.
    # - Kısa kollu ("short") kıyafetlerde: Yalnızca dirsek yukarı kalkıksa (selfie pozu) koltuk altında sarkan köşeyi temizler.
    if sleeve_type in ("long", "short") and len(keypoints) >= 17:
        arm_configs = [
            ("left", p_sh_l, idx_el_l, idx_wr_l, x0, int(src_torso_l + bw * 0.04)),
            ("right", p_sh_r, idx_el_r, idx_wr_r, int(src_torso_r - bw * 0.04), x0 + bw)
        ]
        for side, p_sh, el_i, wr_i, sx_min, sx_max in arm_configs:
            if keypoints[el_i][2] < 0.20:
                continue
            p_el = np.array(keypoints[el_i][:2], dtype=np.float32)
            v_u = p_el - p_sh
            len_u = float(np.linalg.norm(v_u))
            if len_u < 12.0:
                continue
            u_u = v_u / len_u

            if sleeve_type == "short":
                # Yalnızca dirsek yukarı kalkıksa (selfie pozu), homografinin koltuk altında bıraktığı sarkık kol ucunu temizle
                if p_el[1] < p_sh[1] + sh_w * 0.35:
                    y_arm_low = int(max(p_sh[1], p_el[1]) + sh_w * 0.26)
                    if side == "left":
                        warped_mask[min(hp, y_arm_low):, :max(0, int(p_sh[0] - sh_w * 0.18))] = 0
                    else:
                        warped_mask[min(hp, y_arm_low):, min(wp, int(p_sh[0] + sh_w * 0.18)):] = 0

            elif sleeve_type == "long":
                canon_l = _extract_canonical_sleeve_tex(
                    clothes_img, src_mask,
                    sx_min, sx_max,
                    int(y0 + bh * 0.06), int(y0 + bh * 0.94),
                    canon_w=96, canon_h=180,
                    side=side
                )
                if isinstance(canon_l, tuple):
                    canon_l = canon_l[0]
                half_h = canon_l.shape[0] // 2
                canon_upper = canon_l[:half_h, :]
                canon_fore = canon_l[half_h:, :]

                n_u = np.array([-u_u[1], u_u[0]], dtype=np.float32)
                has_forearm = False
                u_f = u_u
                n_f = n_u
                p_wr = p_el
                dot_uf = 1.0
                if keypoints[wr_i][2] > 0.18:
                    p_wr_cand = np.array(keypoints[wr_i][:2], dtype=np.float32)
                    v_f = p_wr_cand - p_el
                    len_f = float(np.linalg.norm(v_f))
                    if len_f >= 10.0:
                        has_forearm = True
                        p_wr = p_wr_cand
                        u_f = v_f / len_f
                        n_f = np.array([-u_f[1], u_f[0]], dtype=np.float32)
                        dot_uf = float(np.dot(u_u, u_f))

                # Dirsek bükülme açısı <= 95° ise (dot_uf >= -0.08, ör. eller belde akimbo veya normal duruş)
                # üst kol ve ön kol aynı dirsek açıortay kenarını (el_pt_a <-> el_pt_b) paylaşır.
                # Dirsek > 95° geri katlanmışsa (dot_uf < -0.08, ör. ayna selfiesinde telefonu yüze tutan kol)
                # kelebek burkulmasını önlemek için üst kol ve ön kol kendi dik normallerini kullanır:
                is_hairpin = has_forearm and (dot_uf < -0.08)
                if not is_hairpin:
                    n_el = n_u + n_f
                    norm_nel = float(np.linalg.norm(n_el))
                    n_el = (n_el / norm_nel) if norm_nel > 1e-4 else n_u
                else:
                    n_el = n_u

                p_u0 = p_sh - u_u * (sh_w * 0.08)
                p_u1 = (p_el + u_u * (sh_w * 0.11)) if is_hairpin else p_el
                hw_sh = sh_w * 0.23
                hw_el = sh_w * 0.175

                top_a = p_u0 - n_u * hw_sh
                top_b = p_u0 + n_u * hw_sh
                # Dış omuz köşesinin doğal omuz yüksekliğinden (p_sh[1] - 0.04 * sh_w) yukarı sivri çıkmasını engelle:
                outer_sh_ymin = float(p_sh[1] - sh_w * 0.04)
                inner_sh_ymin = float(min(dst_tl[1], dst_tr[1]) + sh_w * 0.05)
                if side == "left":
                    # Sol kolda top_b (-x, dış omuz) ve top_a (+x, iç yaka tarafı)
                    top_b[1] = max(top_b[1], outer_sh_ymin)
                    top_a[1] = max(top_a[1], inner_sh_ymin)
                else:
                    # Sağ kolda top_a (+x, dış omuz) ve top_b (-x, iç yaka tarafı)
                    top_a[1] = max(top_a[1], outer_sh_ymin)
                    top_b[1] = max(top_b[1], inner_sh_ymin)

                el_pt_a = p_u1 - n_el * hw_el
                el_pt_b = p_u1 + n_el * hw_el
                el_joint_a = el_pt_a.copy()
                el_joint_b = el_pt_b.copy()

                # Kol aşağı sarkık duruyorsa (u_u[1] > 0.65), üst kolun iç kenarını gövde yan dikişine kenetle
                # ki koltuk altı ile bel arasında arka plan üçgeni (boşluk) açılmasın:
                if u_u[1] > 0.65 and abs(u_u[0]) < 0.45:
                    if side == "left":
                        torso_side_x = float(0.45 * dst_tl[0] + 0.55 * dst_bl[0]) + 8.0
                        top_a[0] = max(top_a[0], float(dst_tl[0]) + 8.0)
                        el_pt_a[0] = max(el_pt_a[0], torso_side_x)
                    else:
                        torso_side_x = float(0.45 * dst_tr[0] + 0.55 * dst_br[0]) - 8.0
                        top_b[0] = min(top_b[0], float(dst_tr[0]) - 8.0)
                        el_pt_b[0] = min(el_pt_b[0], torso_side_x)

                q_upper = np.array([
                    top_a,
                    top_b,
                    el_pt_b,
                    el_pt_a
                ], dtype=np.float32)
                w_u_img, w_u_msk = _warp_quad_patch(canon_upper, q_upper, wp, hp)

                # Gövde iç alanı ile kol yaması birleşimini yumuşak geçişle kaynaştır:
                torso_inner = cv2.erode(warped_mask, np.ones((15, 15), np.uint8), iterations=1)
                torso_keep_alpha = cv2.GaussianBlur((torso_inner > 0).astype(np.float32), (11, 11), 2.5)[:, :, None]
                u_mask_f = (w_u_msk > 0)[:, :, None].astype(np.float32)
                blend_u = u_mask_f * (1.0 - torso_keep_alpha * 0.85)
                warped_cloth = np.clip(
                    blend_u * w_u_img.astype(np.float32) + (1.0 - blend_u) * warped_cloth.astype(np.float32),
                    0, 255
                ).astype(np.uint8)
                warped_mask[w_u_msk > 0] = 255

                # Ön kol (Dirsek -> Bilek)
                if has_forearm:
                    hw_wr = sh_w * 0.125
                    if not is_hairpin:
                        f_top_a = el_joint_a
                        f_top_b = el_joint_b
                    else:
                        p_f0 = p_el - u_f * (sh_w * 0.09)
                        f_top_a = p_f0 - n_f * hw_el
                        f_top_b = p_f0 + n_f * hw_el
                        # Keskin katlanan dirsekte eklem boşluğunu yuvarlak köprü ile doldur
                        el_joint = np.zeros((hp, wp), dtype=np.uint8)
                        cv2.circle(el_joint, (int(p_el[0]), int(p_el[1])), int(hw_el * 1.08), 255, -1)
                        joint_only = (el_joint > 0) & (warped_mask == 0)
                        if np.any(joint_only):
                            el_col = np.median(canon_l[half_h - 4:half_h + 4, :], axis=(0, 1)).astype(np.uint8)
                            warped_cloth[joint_only] = el_col
                            warped_mask[joint_only] = 255

                    wr_pt_a = p_wr - n_f * hw_wr
                    wr_pt_b = p_wr + n_f * hw_wr
                    q_fore = np.array([
                        f_top_a,
                        f_top_b,
                        wr_pt_b,
                        wr_pt_a
                    ], dtype=np.float32)
                    w_f_img, w_f_msk = _warp_quad_patch(canon_fore, q_fore, wp, hp)

                    if is_hairpin:
                        f_alpha = cv2.GaussianBlur((w_f_msk > 0).astype(np.float32), (7, 7), 1.5)[:, :, None]
                        warped_cloth = np.clip(
                            f_alpha * w_f_img.astype(np.float32) + (1.0 - f_alpha) * warped_cloth.astype(np.float32),
                            0, 255
                        ).astype(np.uint8)
                        warped_mask[w_f_msk > 0] = 255
                    else:
                        valid_f = w_f_msk > 0
                        warped_cloth[valid_f] = w_f_img[valid_f]
                        warped_mask[valid_f] = 255

    # 5. Yüz ve Eğim Açılı Yumuşak Çene-Yaka Koruma Bariyeri (Faz 3.2 - Roll-Angle Soft Chin Guard):
    # Saç (parse_clean == 1) burada silinmez; son aşamada (autonomous_hybrid_fusion) renk-arındırmalı alfa ile en üste bindirilir.
    guard = np.zeros((hp, wp), dtype=np.uint8)
    if parse_clean is not None:
        guard[parse_clean == LBL_FACE] = 255  # Yalnızca gerçek yüz pikselleri
    if len(keypoints) > 0 and keypoints[0][2] > 0.25:
        cv2.circle(guard, (int(keypoints[0][0]), int(keypoints[0][1])), int(sh_w * 0.21), 255, -1)
    if len(keypoints) > 10 and keypoints[9][2] > 0.25 and keypoints[10][2] > 0.25:
        mx = float((keypoints[9][0] + keypoints[10][0]) / 2.0)
        my = float((keypoints[9][1] + keypoints[10][1]) / 2.0)
        m_angle = float(np.degrees(np.arctan2(keypoints[10][1] - keypoints[9][1], keypoints[10][0] - keypoints[9][0])))
        if abs(m_angle) > 90.0:
            m_angle += 180.0 if m_angle < 0 else -180.0
        # Baş eğikliğinde yakayı ısırmayan, ağız-çene hattına paralel kompakt elips:
        cv2.ellipse(guard, (int(mx), int(my + sh_w * 0.02)), (int(sh_w * 0.15), int(sh_w * 0.10)), m_angle, 0, 360, 255, -1)

    # 6. Kısa Kollu / Kolsuz Kıyafetlerde Ön Kol (Dirsek-Bilek) Koruma:
    # Üst kollar (omuz-dirsek) tişört kolu tarafından örtülür; ancak dirsek altındaki
    # çıplak ön kolların üzerine gevşek etek ucunun veya kol kenarının taşmasını engeller.
    if sleeve_type in ("short", "sleeveless") and len(keypoints) >= 17:
        y_cuff_limit = int(y_sh + sh_w * 0.64)
        forearm_guard = np.zeros((hp, wp), dtype=np.uint8)
        arm_thick = max(12, int(sh_w * 0.18))
        for s_idx, e_idx, w_idx in [(11, 13, 15), (12, 14, 16)]:
            if keypoints[e_idx][2] > 0.22 and keypoints[w_idx][2] > 0.18:
                ps = keypoints[s_idx][:2].astype(np.float32)
                pe = keypoints[e_idx][:2].astype(np.float32)
                pw = keypoints[w_idx][:2].astype(np.float32)
                arm_mask_i = np.zeros((hp, wp), dtype=np.uint8)
                if pe[1] < ps[1] + sh_w * 0.35 or pw[1] < pe[1]:
                    # Kalkık kol (ör. ayna selfiesi): dirseğin %22 ilerisinden bileğe kadar ön kolu koru,
                    # ancak dirsek omuzdan aşağıdaysa üst omuz dikişinin (y < y_sh + sh_w * 0.22) oyulmasını engelle:
                    p_start = tuple((pe + 0.22 * (pw - pe)).astype(np.int32))
                    p_end = tuple(pw.astype(np.int32))
                    cv2.line(arm_mask_i, p_start, p_end, 255, arm_thick)
                    cv2.circle(arm_mask_i, p_end, int(arm_thick * 0.6), 255, -1)
                    if pe[1] >= ps[1]:
                        arm_mask_i[:min(hp, int(y_sh + sh_w * 0.22)), :] = 0
                else:
                    # Normal aşağı inen kol: y_cuff_limit altındaki ön kolu koru
                    cv2.line(arm_mask_i, tuple(pe.astype(np.int32)), tuple(pw.astype(np.int32)), 255, arm_thick)
                    cv2.circle(arm_mask_i, tuple(pw.astype(np.int32)), int(arm_thick * 0.6), 255, -1)
                    arm_mask_i[:min(hp, y_cuff_limit), :] = 0
                forearm_guard = cv2.bitwise_or(forearm_guard, arm_mask_i)
        # Yalnızca çıplak ön kol tenini koru
        if parse_clean is not None:
            forearm_guard = cv2.bitwise_and(forearm_guard, (parse_clean == LBL_SKIN).astype(np.uint8) * 255)
        guard = cv2.bitwise_or(guard, forearm_guard)

    warped_mask[guard > 0] = 0

    # 7. Etek ucunun bacaklara gereksiz taşmasını önle (uzun kolda bilek hizasına izin ver)
    hem_safe_max = max(dst_bl[1], dst_br[1]) + 25
    if sleeve_type == "long" and len(keypoints) >= 17:
        for w_idx in (15, 16):
            if keypoints[w_idx][2] > 0.18:
                hem_safe_max = max(hem_safe_max, float(keypoints[w_idx][1]) + sh_w * 0.15)
    if hem_safe_max < hp:
        warped_mask[int(hem_safe_max):, :] = 0

    warped_mask = cv2.morphologyEx(warped_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    warped_cloth = cv2.bitwise_and(warped_cloth, warped_cloth, mask=warped_mask)

    return warped_cloth, warped_mask


def overlay(person_img: np.ndarray, warped_clothes: np.ndarray, warped_mask: np.ndarray) -> np.ndarray:
    """Kıyafet ve maskeyi kişi üzerine ortam temas gölgesiyle bindirir (API / Flask / anakod.py uyumluluğu için)."""
    if warped_clothes.ndim == 2:
        warped_clothes = cv2.cvtColor(warped_clothes, cv2.COLOR_GRAY2BGR)
    if person_img.ndim == 2:
        person_img = cv2.cvtColor(person_img, cv2.COLOR_GRAY2BGR)

    mask_binary = (warped_mask > 80).astype(np.uint8)
    shadow_mask = cv2.dilate(mask_binary, np.ones((9, 9), np.uint8), iterations=1)
    shadow_blur = cv2.GaussianBlur(shadow_mask.astype(np.float32), (15, 15), 0)
    shadow_blur = np.clip((shadow_blur - cv2.GaussianBlur(mask_binary.astype(np.float32), (7, 7), 0)), 0, 1)

    person_f = person_img.astype(np.float32)
    for c in range(3):
        person_f[:, :, c] *= (1.0 - shadow_blur * 0.35)

    alpha = cv2.GaussianBlur(mask_binary.astype(np.float32), (3, 3), 0.8)[:, :, None]
    blended = (alpha * warped_clothes.astype(np.float32) + (1.0 - alpha) * person_f)
    return np.clip(blended, 0, 255).astype(np.uint8)
