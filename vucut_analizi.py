"""
vucut_analizi.py - Hassas Vücut ve Kıyafet Anahtar Nokta Analizi

1. BodyParser      : MediaPipe Selfie Multiclass (insan parsing) modeli ile her pikseli
                     sınıflandırır -> 0: arka plan, 1: saç, 2: vücut teni (boyun/kollar),
                     3: yüz, 4: kıyafet, 5: aksesuar.
2. analyze_body    : İskelet noktalarını (MediaPipe Pose) parsing haritasıyla birleştirip
                     gerçek boyun kenarlarını, omuz üst kavisini, omuz dış sınırını,
                     koltuk altını ve kalça hizasını piksel hassasiyetinde bulur.
                     (MediaPipe omuz noktası eklem merkezidir; vücudun dış hattı değildir.)
3. garment_landmarks: Kıyafet maskesinden yaka, omuz, koltuk altı, kol ağzı ve etek
                     noktalarını otomatik çıkarır.
4. tps_warp        : Thin-Plate-Spline (TPS) ile kıyafeti bu nokta eşleşmelerine göre
                     esnek biçimde büker (tek perspektif dönüşümü yerine).
"""
import os
import cv2
import numpy as np

PARSER_MODEL_PATH = os.path.join("models", "selfie_multiclass_256x256.tflite")
PARSER_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
                    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite")

LBL_BG, LBL_HAIR, LBL_SKIN, LBL_FACE, LBL_CLOTH, LBL_ACC = 0, 1, 2, 3, 4, 5


# ─────────────────────────────────────────────────────────────────────────────
# 1. İNSAN PARSING (Saç / Ten / Yüz / Kıyafet)
# ─────────────────────────────────────────────────────────────────────────────
class BodyParser:
    def __init__(self, model_path: str = PARSER_MODEL_PATH, auto_download: bool = True):
        self.available = False
        self.error_message = ""
        self.segmenter = None
        try:
            if not os.path.exists(model_path) and auto_download:
                import urllib.request
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                urllib.request.urlretrieve(PARSER_MODEL_URL, model_path)

            import mediapipe as mp  # noqa: F401
            from mediapipe.tasks import python as mpt
            from mediapipe.tasks.python import vision

            # Türkçe karakterli yollarda sorun yaşamamak için model bellekten yüklenir
            with open(model_path, "rb") as f:
                buf = f.read()
            opts = vision.ImageSegmenterOptions(
                base_options=mpt.BaseOptions(model_asset_buffer=buf),
                output_category_mask=True,
                output_confidence_masks=False,
            )
            self.segmenter = vision.ImageSegmenter.create_from_options(opts)
            self.available = True
        except Exception as exc:
            self.error_message = str(exc)

    def parse(self, image_bgr: np.ndarray):
        """Görsel ile aynı boyutta etiket haritası (uint8) döndürür veya None."""
        if not self.available:
            return None
        try:
            import mediapipe as mp
            rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            res = self.segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            lbl = res.category_mask.numpy_view().squeeze().astype(np.uint8)
            if lbl.shape[:2] != image_bgr.shape[:2]:
                lbl = cv2.resize(lbl, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            return lbl
        except Exception as exc:
            self.error_message = str(exc)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. VÜCUT ANAHTAR NOKTALARI (İskelet + Parsing Füzyonu)
# ─────────────────────────────────────────────────────────────────────────────
def _kp(keypoints, i):
    return np.array(keypoints[i][:2], dtype=np.float32)


def _vis(keypoints, i):
    return float(keypoints[i][2]) if len(keypoints) > i and len(keypoints[i]) > 2 else 0.0


def analyze_body(person_shape, keypoints, parse_map=None):
    """
    Döndürür: dict (tüm noktalar ekran koordinatında; 'l' = ekranın solu, 'r' = ekranın sağı)
    """
    hp, wp = person_shape[:2]
    ls, rs = _kp(keypoints, 11), _kp(keypoints, 12)
    lh, rh = _kp(keypoints, 23), _kp(keypoints, 24)
    le, re = _kp(keypoints, 13), _kp(keypoints, 14)

    # Ekran solu / sağı
    if rs[0] < ls[0]:
        sh_l, sh_r, el_l, el_r = rs, ls, re, le
        vis_el_l, vis_el_r = _vis(keypoints, 14), _vis(keypoints, 13)
    else:
        sh_l, sh_r, el_l, el_r = ls, rs, le, re
        vis_el_l, vis_el_r = _vis(keypoints, 13), _vis(keypoints, 14)
    hip_l, hip_r = (rh, lh) if rh[0] < lh[0] else (lh, rh)

    sh_vec = sh_r - sh_l
    sw = max(20.0, float(np.linalg.norm(sh_vec)))
    sh_unit = sh_vec / sw
    up = np.array([sh_unit[1], -sh_unit[0]], dtype=np.float32)  # omuz hattına dik, yukarı
    if up[1] > 0:
        up = -up
    sh_mid = (sh_l + sh_r) / 2.0
    hip_mid = (hip_l + hip_r) / 2.0

    torso_h = float(np.dot(sh_mid - hip_mid, up))
    hips_ok = (_vis(keypoints, 23) > 0.3 and _vis(keypoints, 24) > 0.3 and sw * 0.7 < torso_h < sw * 2.3)
    if not hips_ok:
        torso_h = sw * 1.35
        hip_l = sh_l - up * torso_h + sh_unit * sw * 0.09
        hip_r = sh_r - up * torso_h - sh_unit * sw * 0.09
        hip_mid = (hip_l + hip_r) / 2.0

    # Ağız / burun (boyun yüksekliği için)
    mouth = (_kp(keypoints, 9) + _kp(keypoints, 10)) / 2.0 if len(keypoints) > 10 else sh_mid + up * sw * 0.55
    if float(np.dot(mouth - sh_mid, up)) < sw * 0.15:
        mouth = sh_mid + up * sw * 0.55
    neck_cx = float(0.5 * sh_mid[0] + 0.5 * mouth[0])

    info = {
        "sw": sw, "sh_unit": sh_unit, "up": up, "torso_h": torso_h,
        "sh_l": sh_l, "sh_r": sh_r, "hip_l": hip_l, "hip_r": hip_r,
        "el_l": el_l, "el_r": el_r, "vis_el_l": vis_el_l, "vis_el_r": vis_el_r,
        "parsed": parse_map is not None,
    }

    # ── Varsayılan (anatomik oranlara dayalı) tahminler ──
    neck_half = sw * 0.19
    neck_base_y = float(sh_mid[1] - sw * 0.13)
    neck_l = np.array([neck_cx - neck_half, neck_base_y + sh_unit[1] * -neck_half], np.float32)
    neck_r = np.array([neck_cx + neck_half, neck_base_y + sh_unit[1] * neck_half], np.float32)
    outer_l = sh_l - sh_unit * sw * 0.13 + up * sw * 0.02
    outer_r = sh_r + sh_unit * sw * 0.13 + up * sw * 0.02

    def shoulder_top_default(x, _nl=neck_l.copy(), _nr=neck_r.copy(), _ol=outer_l.copy(), _or=outer_r.copy(),
                             _nb=neck_base_y):
        # Boyundan omuza doğal iniş eğrisi (varsayılan değerler sabitlenir)
        t_l = np.clip((x - _ol[0]) / max(_nl[0] - _ol[0], 1), 0, 1)
        t_r = np.clip((_or[0] - x) / max(_or[0] - _nr[0], 1), 0, 1)
        if x <= _nl[0]:
            return _ol[1] + (_nl[1] - _ol[1]) * (t_l ** 0.6)
        if x >= _nr[0]:
            return _or[1] + (_nr[1] - _or[1]) * (t_r ** 0.6)
        return _nb

    shoulder_top = shoulder_top_default

    # ── Parsing ile hassas ölçüm ──
    if parse_map is not None and parse_map.shape[:2] == (hp, wp):
        body = np.isin(parse_map, [LBL_SKIN, LBL_CLOTH, LBL_ACC]).astype(np.uint8)
        body = cv2.morphologyEx(body, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        skin = (parse_map == LBL_SKIN).astype(np.uint8)

        # (a) Boyun kenarları: ağız ile omuz hattı arasında, boyun merkezinden dışa doğru ten taraması
        y_top = int(mouth[1] + 0.40 * (sh_mid[1] - mouth[1]))
        y_bot = int(mouth[1] + 0.70 * (sh_mid[1] - mouth[1]))
        lefts, rights = [], []
        cx = int(np.clip(neck_cx, 0, wp - 1))
        for y in range(max(0, y_top), min(hp, y_bot)):
            row = skin[y] | (parse_map[y] == LBL_CLOTH)
            if not row[cx]:
                continue
            xl = cx
            while xl > 0 and row[xl - 1]:
                xl -= 1
            xr = cx
            while xr < wp - 1 and row[xr + 1]:
                xr += 1
            lefts.append(xl)
            rights.append(xr)
        if len(lefts) >= 3:
            nl, nr = float(np.median(lefts)), float(np.median(rights))
            half = (nr - nl) / 2.0
            if sw * 0.12 < half < sw * 0.32:
                neck_cx = (nl + nr) / 2.0
                neck_half = half

        # (b) Omuz üst kavisi: her sütunda vücudun en üst pikseli (saç hariç)
        y_search_top = int(max(0, mouth[1]))

        def col_top(x):
            x = int(np.clip(x, 0, wp - 1))
            y_lim_bot = int(min(hp - 1, sh_mid[1] + sw * 0.35))
            col = body[y_search_top:y_lim_bot, x]
            idx = np.flatnonzero(col)
            if len(idx) == 0:
                return None
            # Saç omzun üzerine düşüyorsa (üstte saç varsa) güvenilmez
            return float(idx[0] + y_search_top)

        # (c) Omuz dış sınırı: omuz noktasının hemen altındaki satırlarda dışa doğru tarama
        def outer_edge(sh_pt, direction):
            best = None
            for dy in np.linspace(-0.02, 0.18, 8):
                y = int(np.clip(sh_pt[1] + dy * sw, 0, hp - 1))
                x0 = int(np.clip(sh_pt[0], 0, wp - 1))
                if not body[y, x0]:
                    continue
                x = x0
                lim = int(sw * 0.34)
                steps = 0
                while 0 < x < wp - 1 and body[y, x + direction] and steps < lim:
                    x += direction
                    steps += 1
                d = steps
                if best is None or d > best[0]:
                    best = (d, x, y)
            return best

        ol = outer_edge(sh_l, -1)
        orr = outer_edge(sh_r, +1)
        if ol is not None and sw * 0.05 < ol[0] < sw * 0.33:
            outer_l = np.array([ol[1] + 2, 0], np.float32)
        else:
            outer_l = np.array([outer_l[0], 0], np.float32)
        if orr is not None and sw * 0.05 < orr[0] < sw * 0.33:
            outer_r = np.array([orr[1] - 2, 0], np.float32)
        else:
            outer_r = np.array([outer_r[0], 0], np.float32)

        # Omuz kavisini sütun sütun örnekle ve makul aralığa sıkıştır
        xs = np.linspace(outer_l[0], outer_r[0], 41)
        tops = []
        for x in xs:
            # Omuz hattının bu x'teki y değeri (eğimli omuz için)
            t = (x - sh_l[0]) / max(sh_r[0] - sh_l[0], 1)
            line_y = sh_l[1] + t * (sh_r[1] - sh_l[1])
            ty = col_top(x)
            lo, hi = line_y - sw * 0.30, line_y + sw * 0.10
            if ty is None or not (lo <= ty <= hi):
                ty = shoulder_top_default(x)
            tops.append(np.clip(ty, lo, hi))
        tops = np.array(tops, np.float32)
        # Hafif yumuşatma
        tops = cv2.GaussianBlur(tops.reshape(1, -1), (1, 5), 0).ravel() if len(tops) >= 5 else tops

        def shoulder_top(x, _xs=xs, _t=tops):
            return float(np.interp(x, _xs, _t))

        outer_l[1] = shoulder_top(outer_l[0] + sw * 0.03)
        outer_r[1] = shoulder_top(outer_r[0] - sw * 0.03)

        nl_x, nr_x = neck_cx - neck_half, neck_cx + neck_half
        # Boynun omuza birleştiği nokta: boyun kenarının biraz dışındaki omuz kavisi
        neck_l = np.array([nl_x, shoulder_top(nl_x - sw * 0.05)], np.float32)
        neck_r = np.array([nr_x, shoulder_top(nr_x + sw * 0.05)], np.float32)
        neck_base_y = float((neck_l[1] + neck_r[1]) / 2.0)

    info.update({
        "neck_cx": float(neck_cx), "neck_half": float(neck_half), "neck_base_y": neck_base_y,
        "neck_l": neck_l, "neck_r": neck_r, "outer_l": outer_l, "outer_r": outer_r,
        "shoulder_top": shoulder_top,
    })

    # Koltuk altı: omuz-kalça hattının ~%30'u, eklem merkezinin hafif dışında
    arm_f = 0.5 * (0.32 * torso_h) + 0.5 * (0.50 * sw)
    info["pit_l"] = sh_l - up * arm_f - sh_unit * sw * 0.03
    info["pit_r"] = sh_r - up * arm_f + sh_unit * sw * 0.03
    return info


# ─────────────────────────────────────────────────────────────────────────────
# 3. KIYAFET ANAHTAR NOKTALARI (Maske Kontur Analizi)
# ─────────────────────────────────────────────────────────────────────────────
def garment_landmarks(mask: np.ndarray):
    m = (mask > 127).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if len(xs) < 200:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    gh = max(y1 - y0, 1)

    left = np.full(m.shape[0], -1, np.int32)
    right = np.full(m.shape[0], -1, np.int32)
    for y in range(y0, y1 + 1):
        idx = np.flatnonzero(m[y])
        if len(idx):
            left[y], right[y] = idx[0], idx[-1]

    def rows(a, b):
        r = np.arange(int(y0 + gh * a), int(y0 + gh * b) + 1)
        return r[left[r] >= 0]

    lower = rows(0.42, 0.85)
    if len(lower) < 3:
        return None
    # Gövde kenarı: kol altından eteğe kadar en dar (bel) bölge -> kollar ve bol etek yanıltmaz
    torso_l = float(np.percentile(left[lower], 75))
    torso_r = float(np.percentile(right[lower], 25))
    tw = max(torso_r - torso_l, 10.0)
    cx = (torso_l + torso_r) / 2.0

    # Üst kontur (her sütunun en üst pikseli)
    top = np.full(m.shape[1], -1, np.int32)
    for x in range(x0, x1 + 1):
        idx = np.flatnonzero(m[:, x])
        if len(idx):
            top[x] = idx[0]

    def top_at(x):
        x = int(np.clip(round(x), x0, x1))
        if top[x] >= 0:
            return float(top[x])
        near = np.flatnonzero(top[max(x0, x - 15):min(x1, x + 15) + 1] >= 0)
        return float(top[max(x0, x - 15) + near[0]]) if len(near) else float(y0)

    # Kollar: gövde genişliğinden belirgin taşan üst satırlar.
    # Koltuk altı = yukarıdan başlayan İLK kol bölgesinin bittiği satır (bol etek yanıltmasın)
    upper = rows(0.0, 0.68)
    sl_flag = left[upper] < torso_l - 0.07 * tw
    sr_flag = right[upper] > torso_r + 0.07 * tw

    def first_run(flags):
        rws, started, gap, last = [], False, 0, None
        for r, f in zip(upper, flags):
            if f:
                started, gap, last = True, 0, r
                rws.append(r)
            elif started:
                gap += 1
                if gap > max(3, int(gh * 0.01)):
                    break
        return np.array(rws, np.int32), last

    sl_rows, last_l = first_run(sl_flag)
    sr_rows, last_r = first_run(sr_flag)
    pit_default = y0 + gh * 0.36
    pit_l_y = float(last_l) if len(sl_rows) > 3 else pit_default
    pit_r_y = float(last_r) if len(sr_rows) > 3 else pit_default
    # Koltuk altı yüksekliği makul değil ise varsayılan
    pit_l_y = pit_l_y if y0 + gh * 0.15 < pit_l_y < y0 + gh * 0.6 else pit_default
    pit_r_y = pit_r_y if y0 + gh * 0.15 < pit_r_y < y0 + gh * 0.6 else pit_default

    def edge_at(y, side):
        y = int(np.clip(y, y0, y1))
        for dy in range(0, 12):
            for yy in (y + dy, y - dy):
                if y0 <= yy <= y1 and left[yy] >= 0:
                    return float(left[yy] if side == "l" else right[yy])
        return torso_l if side == "l" else torso_r

    pit_l = np.array([edge_at(pit_l_y + gh * 0.02, "l"), pit_l_y], np.float32)
    pit_r = np.array([edge_at(pit_r_y + gh * 0.02, "r"), pit_r_y], np.float32)

    # Omuz uçları: gövde kenarı sütununda üst kontur
    sh_l = np.array([min(pit_l[0], torso_l), top_at(min(pit_l[0], torso_l))], np.float32)
    sh_r = np.array([max(pit_r[0], torso_r), top_at(max(pit_r[0], torso_r))], np.float32)

    # Yaka: merkez bölgesinde üst konturdaki en derin çukur ve iki yanındaki en yüksek noktalar
    span = int(0.36 * tw)
    cxi = int(round(cx))
    seg_x = np.arange(max(x0, cxi - span), min(x1, cxi + span) + 1)
    seg_y = np.array([top_at(x) for x in seg_x], np.float32)
    shoulder_level = min(sh_l[1], sh_r[1])
    if len(seg_x) > 10:
        center_band = np.abs(seg_x - cx) < 0.15 * tw
        dip_i = int(np.argmax(np.where(center_band, seg_y, -1)))
        dip = np.array([seg_x[dip_i], seg_y[dip_i]], np.float32)
        lmask, rmask = seg_x < dip[0], seg_x > dip[0]
        cl_i = int(np.argmin(np.where(lmask, seg_y, 1e9)))
        cr_i = int(np.argmin(np.where(rmask, seg_y, 1e9)))
        col_l = np.array([seg_x[cl_i], seg_y[cl_i]], np.float32)
        col_r = np.array([seg_x[cr_i], seg_y[cr_i]], np.float32)
    else:
        dip = col_l = col_r = None

    has_collar = dip is not None and (dip[1] - min(col_l[1], col_r[1])) > gh * 0.025 \
        and (col_r[0] - col_l[0]) > tw * 0.15
    if not has_collar:
        # Düz üst kenar (ör. üstten kırpılmış fotoğraf): yaka genişliğini oranla tahmin et
        col_l = np.array([cx - 0.19 * tw, top_at(cx - 0.19 * tw)], np.float32)
        col_r = np.array([cx + 0.19 * tw, top_at(cx + 0.19 * tw)], np.float32)
        dip = np.array([cx, top_at(cx) + gh * 0.04], np.float32)
        collar_depth = gh * 0.04
    else:
        collar_depth = float(dip[1] - (col_l[1] + col_r[1]) / 2.0)

    # Etek uçları
    hem_rows = rows(0.94, 0.99)
    hy = float(np.median(hem_rows)) if len(hem_rows) else float(y1)
    hem_l = np.array([edge_at(hy, "l"), hy], np.float32)
    hem_r = np.array([edge_at(hy, "r"), hy], np.float32)
    mid_y = (pit_l_y + hy) / 2.0
    waist_l = np.array([edge_at(mid_y, "l"), mid_y], np.float32)
    waist_r = np.array([edge_at(mid_y, "r"), mid_y], np.float32)

    # Kol ağızları (kollar varsa): kolun en dış sütunundaki orta nokta
    def sleeve_end(side):
        rws = sl_rows if side == "l" else sr_rows
        if len(rws) <= 3:
            return None
        if side == "l":
            ext_x = int(left[rws].min())
            if ext_x > torso_l - 0.12 * tw:
                return None
        else:
            ext_x = int(right[rws].max())
            if ext_x < torso_r + 0.12 * tw:
                return None
        col = np.flatnonzero(m[:, ext_x])
        col = col[(col >= y0) & (col <= pit_default + gh * 0.4)]
        if len(col) == 0:
            return None
        return np.array([ext_x, float(np.median(col))], np.float32)

    return {
        "torso_l": torso_l, "torso_r": torso_r, "tw": tw, "cx": cx, "gh": gh, "y0": y0, "y1": y1,
        "col_l": col_l, "col_r": col_r, "dip": dip, "collar_depth": collar_depth, "has_collar": has_collar,
        "sh_l": sh_l, "sh_r": sh_r, "pit_l": pit_l, "pit_r": pit_r,
        "waist_l": waist_l, "waist_r": waist_r, "hem_l": hem_l, "hem_r": hem_r,
        "sleeve_l": sleeve_end("l"), "sleeve_r": sleeve_end("r"),
        "top_at": top_at,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. THIN-PLATE-SPLINE BÜKME
# ─────────────────────────────────────────────────────────────────────────────
def _tps_U(r2):
    with np.errstate(divide="ignore", invalid="ignore"):
        u = r2 * np.log(r2)
    u[~np.isfinite(u)] = 0.0
    return u


def _tps_fit(ctrl, target, reg):
    n = len(ctrl)
    d2 = np.sum((ctrl[:, None, :] - ctrl[None, :, :]) ** 2, axis=2)
    K = _tps_U(d2) + reg * np.eye(n)
    P = np.hstack([np.ones((n, 1)), ctrl])
    L = np.zeros((n + 3, n + 3))
    L[:n, :n], L[:n, n:], L[n:, :n] = K, P, P.T
    Y = np.vstack([target, np.zeros((3, 2))])
    return np.linalg.solve(L, Y)


def _tps_eval(ctrl, params, pts):
    d2 = np.sum((pts[:, None, :] - ctrl[None, :, :]) ** 2, axis=2)
    U = _tps_U(d2)
    n = len(ctrl)
    return U @ params[:n] + params[n] + pts @ params[n + 1:]


def tps_warp(src_img, src_mask, src_pts, dst_pts, out_shape, reg=None, grid_step=4):
    """
    dst (kişi) noktalarını src (kıyafet) noktalarına eşleyen ters TPS haritası kurulur
    ve cv2.remap ile kıyafet kişinin vücut geometrisine bükülür.
    """
    hp, wp = out_shape[:2]
    src_pts = np.asarray(src_pts, np.float64)
    dst_pts = np.asarray(dst_pts, np.float64)
    # Sayısal kararlılık için normalize et
    scale = float(max(hp, wp))
    c = dst_pts / scale
    t = src_pts / scale
    if reg is None:
        reg = 1e-3
    params = _tps_fit(c, t, reg)

    gx = np.arange(0, wp + grid_step, grid_step, dtype=np.float64)
    gy = np.arange(0, hp + grid_step, grid_step, dtype=np.float64)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.stack([GX.ravel(), GY.ravel()], axis=1) / scale
    mapped = _tps_eval(c, params, pts) * scale
    map_x = mapped[:, 0].reshape(GY.shape).astype(np.float32)
    map_y = mapped[:, 1].reshape(GY.shape).astype(np.float32)
    map_x = cv2.resize(map_x, (len(gx) * grid_step, len(gy) * grid_step), interpolation=cv2.INTER_LINEAR)[:hp, :wp]
    map_y = cv2.resize(map_y, (len(gx) * grid_step, len(gy) * grid_step), interpolation=cv2.INTER_LINEAR)[:hp, :wp]
    # resize, grid noktalarını (step-1)/2 piksel sağa/aşağı kaydırır; geri al:
    off = (grid_step - 1) / 2.0
    if off > 0:
        Mshift = np.float32([[1, 0, -off], [0, 1, -off]])
        map_x = cv2.warpAffine(map_x, Mshift, (wp, hp), borderMode=cv2.BORDER_REPLICATE)
        map_y = cv2.warpAffine(map_y, Mshift, (wp, hp), borderMode=cv2.BORDER_REPLICATE)

    warped = cv2.remap(src_img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    warped_m = cv2.remap(src_mask, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, warped_m


# ─────────────────────────────────────────────────────────────────────────────
# 5. GELİŞMİŞ VTON ALTYAPISI (Instance, Kontur, Agnostic ve Örtüşme Koruması)
# ─────────────────────────────────────────────────────────────────────────────
class PersonInstanceSegmenter:
    """
    1. Person Instance Segmentation (YOLO11-Seg):
    Hedef kişiyi tekil olarak arka plandaki çamaşırlardan, perdelerden ve diğer insanlardan ayırır.
    """
    def __init__(self, model_path: str = "models/yolo11n-seg.pt"):
        self.available = False
        self.model = None
        try:
            from ultralytics import YOLO
            if not os.path.exists(model_path):
                model_path = "yolo11n-seg.pt"
            self.model = YOLO(model_path)
            self.available = True
        except Exception as e:
            self.available = False
            self.error = str(e)

    def get_person_mask(self, image_bgr: np.ndarray, keypoints=None) -> np.ndarray:
        hp, wp = image_bgr.shape[:2]
        if not self.available or self.model is None:
            return None

        try:
            results = self.model(image_bgr, classes=[0], verbose=False)
            if not results or len(results[0].boxes) == 0 or results[0].masks is None:
                return None

            boxes = results[0].boxes.xyxy.cpu().numpy()
            masks = results[0].masks.data.cpu().numpy()

            if len(boxes) == 1:
                m = masks[0]
                return cv2.resize((m > 0.5).astype(np.uint8) * 255, (wp, hp), interpolation=cv2.INTER_NEAREST)

            # Birden fazla insan varsa iskelet gövde merkezine en yakın kişiyi seç
            target_idx = 0
            if keypoints is not None and len(keypoints) >= 25:
                tx = float((keypoints[11][0] + keypoints[12][0]) / 2.0)
                ty = float((keypoints[11][1] + keypoints[12][1] + keypoints[23][1] + keypoints[24][1]) / 4.0)
                best_dist = 1e9
                for i, b in enumerate(boxes):
                    cx = (b[0] + b[2]) / 2.0
                    cy = (b[1] + b[3]) / 2.0
                    d = (cx - tx) ** 2 + (cy - ty) ** 2
                    if d < best_dist:
                        best_dist = d
                        target_idx = i

            m = masks[target_idx]
            return cv2.resize((m > 0.5).astype(np.uint8) * 255, (wp, hp), interpolation=cv2.INTER_NEAREST)
        except Exception:
            return None


def refine_person_instance(raw_parse: np.ndarray, keypoints, img_shape, person_instance_mask=None):
    """
    1. Person Instance Segmentation:
    Hedef kişinin gövde iskeletini kapsayan bağlantılı bileşeni izole eder.
    Arka plandaki mobilyaları, askıdaki kıyafetleri veya diğer insanları sıfırlar.
    """
    hp, wp = img_shape[:2]
    if person_instance_mask is not None and person_instance_mask.shape[:2] == (hp, wp):
        instance_mask = (person_instance_mask > 127).astype(np.uint8)
        parse_clean = np.where(instance_mask > 0, raw_parse, 0).astype(np.uint8) if raw_parse is not None else None
        return instance_mask, parse_clean

    if raw_parse is None:
        return np.ones((hp, wp), dtype=np.uint8), None

    body_bin = (raw_parse > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(body_bin, connectivity=8)
    if num_labels <= 1:
        return np.ones((hp, wp), dtype=np.uint8), raw_parse

    # Gövde merkezini bul
    if keypoints is not None and len(keypoints) >= 25:
        torso_x = int(np.clip((keypoints[11][0] + keypoints[12][0]) / 2.0, 0, wp - 1))
        torso_y = int(np.clip((keypoints[11][1] + keypoints[12][1] + keypoints[23][1] + keypoints[24][1]) / 4.0, 0, hp - 1))
        target_label = labels[torso_y, torso_x]
        if target_label == 0:
            target_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    else:
        target_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    instance_mask = (labels == target_label).astype(np.uint8)
    parse_clean = np.where(instance_mask > 0, raw_parse, 0).astype(np.uint8)
    return instance_mask, parse_clean


def extract_body_contours(instance_mask: np.ndarray, keypoints, img_shape):
    """
    2. Body Contour Extraction (Anatomik Güvenlik Kilitli):
    İskelet eklemleri ile vücudun gerçek ten silüet sınırlarını birleştirir.
    Anatomik güvenlik kilidi sayesinde arkaplandaki çamaşır veya kolların
    omuz ve kalçayı çadır gibi dışarı patlatması engellenir.
    """
    hp, wp = img_shape[:2]
    ls = np.array(keypoints[11][:2], dtype=np.float32)
    rs = np.array(keypoints[12][:2], dtype=np.float32)
    lh = np.array(keypoints[23][:2], dtype=np.float32)
    rh = np.array(keypoints[24][:2], dtype=np.float32)

    p_sh_l, p_sh_r = (rs, ls) if rs[0] < ls[0] else (ls, rs)
    p_hip_l, p_hip_r = (rh, lh) if rh[0] < lh[0] else (lh, rh)

    sh_vec = p_sh_r - p_sh_l
    sh_w = max(20.0, float(np.linalg.norm(sh_vec)))
    sh_max_reach = sh_w * 0.30  # Omuz ekleminden en fazla %30 dışarı ten uzanabilir
    hip_max_reach = sh_w * 0.35 # Kalça ekleminden en fazla %35 dışarı kalça uzanabilir

    y_sh = int((p_sh_l[1] + p_sh_r[1]) / 2.0)
    y_hip = int((p_hip_l[1] + p_hip_r[1]) / 2.0)

    # Omuz seviyesinde fiziksel kontur taraması
    raw_sh_l = p_sh_l[0] - 20.0
    raw_sh_r = p_sh_r[0] + 20.0
    if instance_mask is not None:
        sh_scan_y = int(np.clip(y_sh, 0, hp - 1))
        sh_rows = np.where(instance_mask[sh_scan_y, :] > 0)[0]
        if len(sh_rows) > 10:
            raw_sh_l = float(sh_rows.min())
            raw_sh_r = float(sh_rows.max())

    contour_sh_l_x = float(np.clip(raw_sh_l, p_sh_l[0] - sh_max_reach, p_sh_l[0] + 5.0))
    contour_sh_r_x = float(np.clip(raw_sh_r, p_sh_r[0] - 5.0, p_sh_r[0] + sh_max_reach))

    # Kalça seviyesinde fiziksel kontur taraması
    raw_hip_l = p_hip_l[0] - 15.0
    raw_hip_r = p_hip_r[0] + 15.0
    if instance_mask is not None:
        hip_scan_y = int(np.clip(y_hip, 0, hp - 1))
        hip_rows = np.where(instance_mask[hip_scan_y, :] > 0)[0]
        if len(hip_rows) > 10:
            raw_hip_l = float(hip_rows.min())
            raw_hip_r = float(hip_rows.max())

    contour_hip_l_x = float(np.clip(raw_hip_l, p_hip_l[0] - hip_max_reach, p_hip_l[0] + 5.0))
    contour_hip_r_x = float(np.clip(raw_hip_r, p_hip_r[0] - 5.0, p_hip_r[0] + hip_max_reach))

    # Bel oyuntusu taraması (omuz ile kalça arası en dar bölge)
    waist_min_w = wp
    waist_l_x, waist_r_x = p_sh_l[0], p_sh_r[0]
    waist_y = int(y_sh + (y_hip - y_sh) * 0.5)
    if instance_mask is not None and y_hip > y_sh + 20:
        for y in range(int(y_sh + (y_hip - y_sh) * 0.35), int(y_sh + (y_hip - y_sh) * 0.75)):
            r = np.where(instance_mask[y, :] > 0)[0]
            if len(r) > 10 and (r.max() - r.min()) < waist_min_w:
                waist_min_w = r.max() - r.min()
                waist_l_x = float(r.min())
                waist_r_x = float(r.max())
                waist_y = y

    return {
        "sh_l_x": contour_sh_l_x,
        "sh_r_x": contour_sh_r_x,
        "sh_w": max(30.0, contour_sh_r_x - contour_sh_l_x),
        "hip_l_x": contour_hip_l_x,
        "hip_r_x": contour_hip_r_x,
        "waist_l_x": waist_l_x,
        "waist_r_x": waist_r_x,
        "waist_y": waist_y,
    }


def create_occlusion_mask(parse_map: np.ndarray, instance_mask: np.ndarray, keypoints, armpit_y: float):
    """
    Doğal Örtüşme (Occlusion) Koruması (Sprint 2 Baseline Uyumlu):
    - Saçlar: Omuz ve göğse dökülen tüm saç telleri (LBL_HAIR) kıyafetin önünde korunur.
    - Eller ve Bilekler: Göğüs/karın üzerine gelen eller (keypoints 15-22) öne alınır.
    - Üst kollar KESİNLİKLE delinmez: Yeni t-shirt kolu üst kol tenini doğal olarak örtmelidir.
    """
    if instance_mask is not None:
        hp, wp = instance_mask.shape[:2]
    elif parse_map is not None:
        hp, wp = parse_map.shape[:2]
    else:
        return None
    valid_instance = instance_mask if instance_mask is not None else np.ones((hp, wp), dtype=np.uint8)

    # 1. Saç (parse == 1) - Omuz ve göğüs bölgesine dökülen saç telleri
    hair_occ = ((parse_map == LBL_HAIR) & (valid_instance > 0)).astype(np.uint8) * 255

    # 2. Eller ve Parmaklar (Yalnızca gövde/göğüs üzerine konan eller)
    hand_occ = np.zeros((hp, wp), dtype=np.uint8)
    if keypoints is not None and len(keypoints) >= 17:
        hand_mask = np.zeros((hp, wp), dtype=np.uint8)
        try:
            sh_w = float(np.linalg.norm(np.array(keypoints[12][:2]) - np.array(keypoints[11][:2])))
        except Exception:
            sh_w = wp * 0.35
        hand_radius = int(np.clip(sh_w * 0.12, 12, 35))

        # Yalnızca bilek ve parmak eklemleri
        hand_indices = [15, 16]
        if len(keypoints) >= 23:
            hand_indices += [17, 18, 19, 20, 21, 22]

        for h_idx in hand_indices:
            if h_idx < len(keypoints) and keypoints[h_idx][2] > 0.35:
                hx, hy = int(keypoints[h_idx][0]), int(keypoints[h_idx][1])
                # Yalnızca gövde sınırları içinde kalan eller
                if 0 <= hx < wp and 0 <= hy < hp:
                    cv2.circle(hand_mask, (hx, hy), hand_radius, 255, -1)

        if np.any(hand_mask > 0):
            hand_occ = ((hand_mask > 0) & (parse_map == LBL_SKIN) & (valid_instance > 0)).astype(np.uint8) * 255

    occlusion = cv2.bitwise_or(hair_occ, hand_occ)
    return occlusion


def restore_neck_skin(person_bgr: np.ndarray, exposed_old_mask: np.ndarray, parse_map: np.ndarray, y_shoulder: float, instance_mask: np.ndarray = None, hem_y_max: int = None):
    """
    3. Garment-Agnostic Mask Refinement (Akıllı Yaka & Ten Rekonstrüksiyonu):
    - Eski kıyafet silinirken boyun çukurunda kalan boşlukları yüz/boyun ten tonu degrade gradyanı ile doğal tamamlar.
    - Alt beden bariyeri: Etek ucu (hem_y_max) altındaki hiçbir piksel (pantolon/eşofman) inpaint edilmez.
    - Arka plan koruma bariyeri: İnterpolasyon yalnızca kişi silüeti içinde kalır, arka plana taşmaz.
    - Sınır dikişi dikiş kaynağı: Bütün boynu silip bulanıklaştırmak yerine yalnızca 2-3 piksellik kenar dikişini telea ile harmanlar.
    """
    if exposed_old_mask is None or not np.any(exposed_old_mask > 0):
        return person_bgr

    hp, wp = person_bgr.shape[:2]
    clean_person = person_bgr.copy()
    valid_mask = (exposed_old_mask > 0)

    # 1. Alt Beden Bariyeri (Etek ucunun altındaki eşofman/pantolonu koru)
    if hem_y_max is not None and hem_y_max > 0:
        valid_mask[hem_y_max + 2:, :] = False

    # 2. Arka Plan Bariyeri (Kişi instance sınırının dışına taşmayı engelle)
    inst_safe = None
    if instance_mask is not None:
        inst_bin = (instance_mask > 0).astype(np.uint8)
        inst_safe = cv2.dilate(inst_bin, np.ones((3, 3), np.uint8), iterations=1)
        valid_mask = valid_mask & (inst_safe > 0)

    if not np.any(valid_mask):
        return person_bgr

    # 3. Boyun / Omuz Ten Rengi Örnekleme (Gözlük ve gölge piksellerinden etkilenmemek için aydınlık ten piksellerini seç)
    skin_bgr = np.array([150, 170, 200], dtype=np.float32)
    chin_color = None
    if parse_map is not None:
        y_grid_all = np.arange(hp)[:, None]
        neck_skin_mask = (parse_map == LBL_SKIN) & (y_grid_all < (y_shoulder + 35))
        if np.sum(neck_skin_mask) > 30:
            ns_pixels = person_bgr[neck_skin_mask]
            luma = 0.114 * ns_pixels[:, 0] + 0.587 * ns_pixels[:, 1] + 0.299 * ns_pixels[:, 2]
            p40, p90 = np.percentile(luma, [40, 90])
            bright_ns = ns_pixels[(luma >= p40) & (luma <= p90)]
            if len(bright_ns) > 10:
                chin_color = np.median(bright_ns, axis=0).astype(np.float32)

        if chin_color is None:
            face_pts = np.where(parse_map == LBL_FACE)
            if len(face_pts[0]) > 50:
                face_pixels = person_bgr[face_pts]
                luma_f = 0.114 * face_pixels[:, 0] + 0.587 * face_pixels[:, 1] + 0.299 * face_pixels[:, 2]
                p50, p90 = np.percentile(luma_f, [50, 90])
                bright_f = face_pixels[(luma_f >= p50) & (luma_f <= p90)]
                if len(bright_f) > 10:
                    chin_color = np.median(bright_f, axis=0).astype(np.float32)
                else:
                    chin_color = np.median(face_pixels, axis=0).astype(np.float32)
    if chin_color is None:
        chin_color = skin_bgr

    # 4. Merkez Yaka Çukuru ve Omuz Askıları / Yan Kenar Temizliği
    y_grid = np.arange(hp)[:, None]
    x_grid = np.arange(wp)[None, :]
    # Merkez boyun çukuru (yalnızca boyun ortası)
    neck_cx = wp * 0.5
    if parse_map is not None:
        f_xs = np.where(parse_map == LBL_FACE)[1]
        if len(f_xs) > 20:
            neck_cx = float(np.median(f_xs))
    central_neck = (np.abs(x_grid - neck_cx) < (wp * 0.11)) & (y_grid >= (y_shoulder - 22)) & (y_grid < (y_shoulder + 25))
    neck_zone = valid_mask & central_neck

    if np.any(neck_zone):
        neck_ys = np.where(neck_zone)[0]
        y_min_nz, y_max_nz = neck_ys.min(), neck_ys.max()
        h_nz = max(y_max_nz - y_min_nz, 1)

        t = np.clip((y_grid - y_min_nz) / float(h_nz), 0.0, 1.0)
        grad_1d = chin_color * (1.0 - t * 0.08)
        neck_canvas = np.empty((hp, wp, 3), dtype=np.uint8)
        neck_canvas[:] = np.clip(grad_1d, 0, 255).astype(np.uint8)[:, None, :]

        clean_person[neck_zone] = neck_canvas[neck_zone]

        nz_u8 = neck_zone.astype(np.uint8) * 255
        nz_dil = cv2.dilate(nz_u8, np.ones((3, 3), np.uint8), iterations=1)
        nz_ero = cv2.erode(nz_u8, np.ones((3, 3), np.uint8), iterations=1)
        seam_mask = cv2.subtract(nz_dil, nz_ero)

        if np.any(seam_mask > 0):
            clean_person = cv2.inpaint(clean_person, seam_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # 5. Omuz Askıları, Boyun İpleri (Halter Strap) ve Gövde Kenarında Kalan Eski Kıyafet Parçaları
    other_zone = valid_mask & (~neck_zone)
    if hem_y_max is None or hem_y_max <= 0:
        other_zone[int(y_shoulder + 180):, :] = False

    if np.any(other_zone):
        other_u8 = other_zone.astype(np.uint8) * 255
        other_dil = cv2.dilate(other_u8, np.ones((7, 7), np.uint8), iterations=1)
        if parse_map is not None:
            other_dil[np.isin(parse_map, [LBL_HAIR, LBL_FACE])] = 0
        if inst_safe is not None:
            other_dil = cv2.bitwise_and(other_dil, inst_safe * 255)

        if np.any(other_dil > 0):
            # Referans tuvalde (ref_canvas) komşu eski kıyafet (LBL_CLOTH) piksellerinin inpaint sırasında
            # dışarıya (ör. bel yanındaki kırmızı tişört çıkıntısına) geri bulaşmasını %100 engelle:
            # - Üst boyun bölgesinde (y < y_shoulder + 20) saç ve kıyafet komşularını chin_color ile maskele,
            # - Alt gövde/bel yanlarında (y >= y_shoulder + 20) ise eski kıyafet içini arka plan rengiyle (env_col) maskele:
            ref_canvas = clean_person.copy()
            if parse_map is not None:
                neck_band_non_skin = np.isin(parse_map, [LBL_HAIR, LBL_CLOTH]) & (y_grid < int(y_shoulder + 20))
                ref_canvas[neck_band_non_skin & (other_dil == 0)] = np.clip(chin_color, 0, 255).astype(np.uint8)

                lower_cloth_ref = (parse_map == LBL_CLOTH) & (other_dil == 0) & (y_grid >= int(y_shoulder + 20))
                if np.any(lower_cloth_ref):
                    bg_zone = (y_grid >= int(y_shoulder + 20)) & ((inst_safe == 0) if inst_safe is not None else (parse_map == 0))
                    if np.sum(bg_zone) > 50:
                        env_col = np.median(clean_person[bg_zone], axis=0).astype(np.uint8)
                        ref_canvas[lower_cloth_ref] = env_col

            inpainted = cv2.inpaint(ref_canvas, other_dil, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
            alpha_inpaint = cv2.GaussianBlur((other_dil > 0).astype(np.float32), (5, 5), 0.8)[:, :, None]
            clean_person = np.clip(
                alpha_inpaint * inpainted.astype(np.float32) + (1.0 - alpha_inpaint) * clean_person.astype(np.float32),
                0, 255
            ).astype(np.uint8)

    return clean_person
