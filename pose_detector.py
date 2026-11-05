"""
Human pose estimation wrapper using MediaPipe Pose and a built-in
biomechanical kinematic simulator for camera-free testing and validation.
"""

import math
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

from config import LANDMARK_INDICES

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class PoseDetector:
    """Extracts normalized 2D body landmarks from RGB video frames using MediaPipe Pose."""

    def __init__(
        self,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        self.pose = None
        if MEDIAPIPE_AVAILABLE and hasattr(mp, "solutions"):
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

    def extract_landmarks(
        self, frame_bgr: np.ndarray
    ) -> Optional[Dict[str, Tuple[float, float, float]]]:
        """Processes a BGR frame and returns a dictionary of normalized (x, y, visibility) coordinates."""
        if self.pose is None:
            return None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if not results.pose_landmarks:
            return None

        lm_list = results.pose_landmarks.landmark
        extracted: Dict[str, Tuple[float, float, float]] = {}

        for name, idx in LANDMARK_INDICES.items():
            lm = lm_list[idx]
            extracted[name] = (float(lm.x), float(lm.y), float(lm.visibility))

        return extracted

    def close(self) -> None:
        """Releases MediaPipe resources."""
        if self.pose is not None:
            self.pose.close()


class SyntheticBiomechanicsSimulator:
    """
    Generates realistic human movement sequences (Standing, Lateral Leaning,
    Squatting, Single-Leg Stance, and Boundary Excursion) for simulation mode.
    """

    @staticmethod
    def generate_frame_and_landmarks(
        timestamp_s: float, width: int = 1280, height: int = 720
    ) -> Tuple[np.ndarray, Dict[str, Tuple[float, float, float]], str]:
        """Synthesizes a studio background canvas, normalized landmarks, and posture phase label."""
        canvas = np.full((height, width, 3), (28, 30, 36), dtype=np.uint8)

        # Draw subtle biomechanical calibration grid
        grid_step = 60
        for x in range(0, width, grid_step):
            cv2.line(canvas, (x, 0), (x, height), (38, 40, 48), 1)
        for y in range(0, height, grid_step):
            cv2.line(canvas, (0, y), (width, y), (38, 40, 48), 1)

        # Cycle through 5 distinct biomechanical test phases (20-second loop)
        cycle_t = timestamp_s % 20.0
        micro_sway = 0.003 * math.sin(timestamp_s * 3.0)

        lean_x = 0.0
        squat_depth = 0.0
        left_foot_lift = 0.0

        if cycle_t < 4.0:
            phase_label = "Quiet Standing"
            lean_x = micro_sway
        elif cycle_t < 8.0:
            phase_label = "Lateral Lean (Within BOS)"
            alpha = math.sin(((cycle_t - 4.0) / 4.0) * math.pi)
            lean_x = 0.052 * alpha + micro_sway
        elif cycle_t < 12.0:
            phase_label = "Deep Squatting"
            alpha = math.sin(((cycle_t - 8.0) / 4.0) * math.pi)
            squat_depth = 0.135 * alpha
            lean_x = micro_sway * 1.2
        elif cycle_t < 16.0:
            phase_label = "Single-Leg Balance"
            alpha = math.sin(((cycle_t - 12.0) / 4.0) * math.pi)
            left_foot_lift = 0.080 * alpha
            # Shift torso over the supporting right leg (-X) to maintain single-leg balance
            lean_x = -0.072 * alpha + micro_sway
        else:
            phase_label = "Extreme Lean (Unstable)"
            alpha = math.sin(((cycle_t - 16.0) / 4.0) * math.pi)
            # Excursion beyond BOS boundary to trigger UNSTABLE fall-risk state
            lean_x = 0.125 * alpha + micro_sway

        cx = 0.50
        ground_y = 0.84

        # Construct anatomically proportioned 2D landmarks
        hip_x = cx + lean_x * 0.75
        hip_y = 0.52 + squat_depth

        shoulder_x = cx + lean_x * 1.18
        shoulder_y = 0.28 + squat_depth * 0.88

        knee_out = squat_depth * 0.35
        landmarks: Dict[str, Tuple[float, float, float]] = {
            "nose": (shoulder_x + lean_x * 0.15, shoulder_y - 0.10, 0.99),
            "left_shoulder": (shoulder_x + 0.065, shoulder_y, 0.99),
            "right_shoulder": (shoulder_x - 0.065, shoulder_y, 0.99),
            "left_elbow": (shoulder_x + 0.085 + lean_x * 0.20, shoulder_y + 0.12, 0.99),
            "right_elbow": (shoulder_x - 0.085 + lean_x * 0.20, shoulder_y + 0.12, 0.99),
            "left_wrist": (shoulder_x + 0.095 + lean_x * 0.30, shoulder_y + 0.23, 0.99),
            "right_wrist": (shoulder_x - 0.095 + lean_x * 0.30, shoulder_y + 0.23, 0.99),
            "left_hip": (hip_x + 0.042, hip_y, 0.99),
            "right_hip": (hip_x - 0.042, hip_y, 0.99),
            "left_knee": (
                cx + 0.048 + knee_out + lean_x * 0.25,
                0.68 + squat_depth * 0.35 - left_foot_lift * 0.55,
                0.99,
            ),
            "right_knee": (
                cx - 0.048 - knee_out + lean_x * 0.25,
                0.68 + squat_depth * 0.35,
                0.99,
            ),
            "left_ankle": (cx + 0.048, ground_y - 0.02 - left_foot_lift, 0.99),
            "right_ankle": (cx - 0.048, ground_y - 0.02, 0.99),
            "left_heel": (cx + 0.038, ground_y - left_foot_lift, 0.99),
            "right_heel": (cx - 0.038, ground_y, 0.99),
            "left_foot_index": (cx + 0.068, ground_y - left_foot_lift, 0.99),
            "right_foot_index": (cx - 0.068, ground_y, 0.99),
        }

        return canvas, landmarks, phase_label