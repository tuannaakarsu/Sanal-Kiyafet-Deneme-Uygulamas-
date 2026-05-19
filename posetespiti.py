import cv2
import mediapipe as mp
import numpy as np


class PoseEstimator:
    def __init__(self, static_image_mode=True):
        self.mp_pose = None
        self.pose = None
        self.mp_drawing = None
        self.available = False
        self.error_message = ""

        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(static_image_mode=static_image_mode)
            self.mp_drawing = mp.solutions.drawing_utils
            self.available = True
        except Exception as exc:
            self.error_message = str(exc)

    def get_keypoints(self, image):
        if not self.available or self.pose is None:
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        if not results.pose_landmarks:
            return None

        h, w, _ = image.shape
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.append([int(lm.x * w), int(lm.y * h), lm.visibility])
        return np.array(keypoints), results.pose_landmarks

    def draw_pose(self, image, landmarks=None):
        if not self.available or self.pose is None or self.mp_drawing is None:
            return image

        if landmarks is None:
            results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            landmarks = results.pose_landmarks
        if landmarks:
            self.mp_drawing.draw_landmarks(image, landmarks, self.mp_pose.POSE_CONNECTIONS)
        return image

    def render_openpose_map(self, image, thickness=6, joint_radius=5):
        """
        Siyah fonda OpenPose-benzeri renkli iskelet haritasi uretir.
        MediaPipe keypointlerini kullanir.
        """
        h, w = image.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
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
