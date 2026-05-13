import cv2
import mediapipe as mp
import numpy as np

class PoseEstimator:
    def __init__(self, static_image_mode=True):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=static_image_mode)
        self.mp_drawing = mp.solutions.drawing_utils

    def get_keypoints(self, image):

        """KEYPOİNTS"""

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

        if landmarks is None:
            results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            landmarks = results.pose_landmarks
        if landmarks:
            self.mp_drawing.draw_landmarks(image, landmarks, self.mp_pose.POSE_CONNECTIONS)
        return image

    def visualize_keypoints(self, image, keypoints, point_names=None):

        image_copy = image.copy()
        important_indices = [11, 12, 23, 24]  # LS, RS, LH, RH
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
        names = ['LS', 'RS', 'LH', 'RH'] if point_names is None else point_names

        for idx, color, name in zip(important_indices, colors, names):
            if keypoints is None or len(keypoints) <= idx:
                continue
            x, y = int(keypoints[idx][0]), int(keypoints[idx][1])
            cv2.circle(image_copy, (x, y), 8, color, -1)
            cv2.putText(image_copy, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return image_copy
