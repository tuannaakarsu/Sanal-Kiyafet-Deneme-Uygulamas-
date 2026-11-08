import os
import cv2
import numpy as np


# Windows üzerinde klasör yolunda Türkçe/özel karakter ("ı", "ş" vb.) varsa
# MediaPipe C++ binarypb dosyasını bulamaz ve hata verir.
# Bunu 8.3 kısa yol (short path) ile çözüyoruz:
if os.name == "nt":
    try:
        import ctypes
        import mediapipe.python.solution_base as _sb
        if not getattr(_sb.SolutionBase, "_short_path_patched", False):
            _orig_sb_init = _sb.SolutionBase.__init__
            def _safe_sb_init(self, binary_graph_path=None, graph_config=None, *args, **kwargs):
                root_path = os.sep.join(os.path.abspath(_sb.__file__).split(os.sep)[:-3])
                buf = ctypes.create_unicode_buffer(500)
                if ctypes.windll.kernel32.GetShortPathNameW(root_path, buf, 500) > 0:
                    from mediapipe.python._framework_bindings import resource_util
                    resource_util.set_resource_dir(buf.value)
                return _orig_sb_init(self, binary_graph_path=binary_graph_path, graph_config=graph_config, *args, **kwargs)
            _sb.SolutionBase.__init__ = _safe_sb_init
            _sb.SolutionBase._short_path_patched = True
    except Exception:
        pass

import mediapipe as mp


class PoseEstimator:
    def __init__(self, static_image_mode=True):
        self.mp_pose = None
        self.pose = None
        self.face_detector = None
        self.mp_drawing = None
        self.available = False
        self.error_message = ""

        try:
            self.mp_pose = mp.solutions.pose
            # Sprint 2 Baseline uyumlu model_complexity=1: Omuz ve kalça eklemlerinde
            # en dengeli ve simetrik sonuçları üretir (aşırı uç sapmaları önler).
            self.pose = self.mp_pose.Pose(static_image_mode=static_image_mode, model_complexity=1)
            if hasattr(mp.solutions, "face_detection"):
                self.face_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.4
                )
            self.mp_drawing = mp.solutions.drawing_utils
            self.available = True
        except Exception as exc:
            self.error_message = str(exc)

    def get_keypoints(self, image):
        if not self.available or self.pose is None:
            return None

        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 1. Öncelikle tam görsel üzerinde iskelet tespiti (kırpma kaynaklı omuz/kalça kaymasını önler)
        kps_full, lms_full = self._process_tta(image_rgb)

        # 2. Kalabalık arka plan kontrolü: Eğer görselde belirgin bir ön plan yüzü varsa
        # ve tam görsel iskeleti bu yüzden çok uzaktaki bir arka plan kişisine kilitlendiyse kırparak odaklan
        if self.face_detector is not None:
            try:
                face_res = self.face_detector.process(image_rgb)
                if face_res.detections and len(face_res.detections) > 1:
                    best_face = None
                    max_area = 0.0
                    for det in face_res.detections:
                        bb = det.location_data.relative_bounding_box
                        area = (bb.width * w) * (bb.height * h)
                        if area > max_area:
                            max_area = area
                            best_face = (int(bb.xmin * w), int(bb.ymin * h), int(bb.width * w), int(bb.height * h))

                    if best_face is not None and max_area > (w * h * 0.003):
                        fx, fy, fw, fh = best_face
                        fcx, fcy = fx + fw * 0.5, fy + fh * 0.5
                        need_crop = (kps_full is None) or (
                            len(kps_full) > 0 and np.hypot(kps_full[0, 0] - fcx, kps_full[0, 1] - fcy) > fw * 1.6
                        )
                        if need_crop:
                            crop_x1 = max(0, int(fcx - fw * 3.0))
                            crop_x2 = min(w, int(fcx + fw * 3.0))
                            crop_y1 = max(0, int(fy - fh * 0.8))
                            crop_y2 = h  # Alt gövdeyi ve kalçayı ASLA kesme
                            crop = image_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
                            if crop.shape[0] > 20 and crop.shape[1] > 20:
                                kps_c, lms_c = self._process_tta(crop)
                                if kps_c is not None:
                                    kps_c[:, 0] += crop_x1
                                    kps_c[:, 1] += crop_y1
                                    if lms_c is not None:
                                        for idx_lm, lm in enumerate(lms_c.landmark):
                                            lm.x = float(kps_c[idx_lm, 0]) / float(w)
                                            lm.y = float(kps_c[idx_lm, 1]) / float(h)
                                    return kps_c, lms_c
            except Exception:
                pass

        if kps_full is None:
            return None
        return kps_full, lms_full

    # MediaPipe 33 nokta için sol/sağ ayna eşleşmeleri (yatay çevirme TTA'sı için)
    _FLIP_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
                   (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]

    def _landmarks_to_array(self, landmarks, w, h):
        return np.array([[lm.x * w, lm.y * h, lm.visibility] for lm in landmarks.landmark], dtype=np.float32)

    def _process_tta(self, image_rgb):
        """
        Test-Time Augmentation (TTA): Görüntü hem normal hem yatay çevrilmiş halde
        iskelet modelinden geçirilir; iki tahmin görünürlük ağırlıklı ortalanır.
        Bu, omuz/kalça noktalarındaki titremeyi ve birkaç piksellik sapmayı belirgin şekilde azaltır.
        Döndürür: (keypoints float32 [33,3] (x, y, visibility), pose_landmarks) veya (None, None)
        """
        h, w = image_rgb.shape[:2]
        res = self.pose.process(image_rgb)
        if not res.pose_landmarks:
            return None, None
        kps = self._landmarks_to_array(res.pose_landmarks, w, h)

        try:
            res_f = self.pose.process(np.ascontiguousarray(image_rgb[:, ::-1]))
            if res_f.pose_landmarks:
                kf = self._landmarks_to_array(res_f.pose_landmarks, w, h)
                kf[:, 0] = (w - 1) - kf[:, 0]
                for a, b in self._FLIP_PAIRS:
                    kf[[a, b]] = kf[[b, a]]
                # İki tahmin birbirinden çok farklıysa (yanlış eşleşme) TTA'yı uygulama
                torso = [11, 12, 23, 24]
                sw = max(np.linalg.norm(kps[11, :2] - kps[12, :2]), 1.0)
                if np.mean(np.linalg.norm(kps[torso, :2] - kf[torso, :2], axis=1)) < 0.25 * sw:
                    wa = np.clip(kps[:, 2:3], 0.05, 1.0)
                    wb = np.clip(kf[:, 2:3], 0.05, 1.0)
                    kps[:, :2] = (kps[:, :2] * wa + kf[:, :2] * wb) / (wa + wb)
                    kps[:, 2] = np.maximum(kps[:, 2], kf[:, 2])
        except Exception:
            pass
        return kps, res.pose_landmarks

    def draw_pose(self, image, landmarks=None):
        if not self.available or self.pose is None or self.mp_drawing is None:
            return image

        if landmarks is None:
            results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            landmarks = results.pose_landmarks
        if landmarks:
            self.mp_drawing.draw_landmarks(image, landmarks, self.mp_pose.POSE_CONNECTIONS)
        return image

    def render_openpose_map(self, image, thickness=6, joint_radius=5, keypoints=None):
        """
        Siyah fonda OpenPose-benzeri renkli iskelet haritasi uretir.
        MediaPipe keypointlerini kullanir.
        """
        h, w = image.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        if keypoints is None:
            keypoints_and_landmarks = self.get_keypoints(image)
            if keypoints_and_landmarks is None:
                return canvas
            keypoints, _ = keypoints_and_landmarks

        # MediaPipe Pose indexleri ile ust-govde + alt-govde zincirleri.
        edges = [
            (0, 11), (0, 12),           # nose -> shoulders
            (11, 12),                   # shoulder line
            (11, 13), (13, 15),         # left arm
            (12, 14), (14, 16),         # right arm
            (11, 23), (12, 24),         # torso sides
            (23, 24),                   # hip line
            (23, 25), (25, 27),         # left leg
            (24, 26), (26, 28),         # right leg
        ]

        # OpenPose benzeri renk paleti (BGR).
        palette = [
            (255, 0, 255), (255, 80, 0), (255, 160, 0), (255, 255, 0),
            (160, 255, 0), (0, 255, 0), (0, 255, 120), (0, 255, 255),
            (0, 180, 255), (0, 100, 255), (0, 0, 255), (120, 0, 255),
        ]

        for i, (a, b) in enumerate(edges):
            if len(keypoints) <= max(a, b):
                continue
            xa, ya, va = keypoints[a]
            xb, yb, vb = keypoints[b]
            if va < 0.2 or vb < 0.2:
                continue
            p1 = (int(xa), int(ya))
            p2 = (int(xb), int(yb))
            color = palette[i % len(palette)]
            cv2.line(canvas, p1, p2, color, thickness, lineType=cv2.LINE_AA)

        for i in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            if len(keypoints) <= i:
                continue
            x, y, v = keypoints[i]
            if v < 0.2:
                continue
            color = palette[i % len(palette)]
            cv2.circle(canvas, (int(x), int(y)), joint_radius, color, -1, lineType=cv2.LINE_AA)

        return canvas

    def visualize_keypoints(self, image, keypoints, point_names=None):
        image_copy = image.copy()
        important_indices = [11, 12, 23, 24]  # LS, RS, LH, RH
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
        names = ["LS", "RS", "LH", "RH"] if point_names is None else point_names

        for idx, color, name in zip(important_indices, colors, names):
            if keypoints is None or len(keypoints) <= idx:
                continue
            x, y = int(keypoints[idx][0]), int(keypoints[idx][1])
            cv2.circle(image_copy, (x, y), 8, color, -1)
            cv2.putText(image_copy, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return image_copy
