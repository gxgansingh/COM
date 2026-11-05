"""
Core biomechanical calculations for segment COM, total body COM,
Base of Support (BOS), Margin of Stability (MoS), and postural sway kinematics.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np

from config import (
    ANTHROPOMETRIC_SEGMENTS,
    DEFAULT_SUBJECT_HEIGHT_CM,
    SINGLE_LEG_LIFT_THRESHOLD_RATIO,
    STABILITY_WARNING_MARGIN_RATIO,
)


@dataclass
class SegmentCOMResult:
    """Stores calculated coordinates and weight for an individual body segment."""
    name: str
    proximal_px: Tuple[int, int]
    distal_px: Tuple[int, int]
    com_px: Tuple[int, int]
    com_norm: Tuple[float, float]
    mass_fraction: float


@dataclass
class PosturalTelemetry:
    """Stores complete frame-by-frame biomechanical stability metrics."""
    timestamp_s: float
    frame_idx: int
    com_px: Tuple[int, int]
    com_norm: Tuple[float, float]
    com_cm: Tuple[float, float]
    com_displacement_cm: float
    sway_velocity_cm_s: float
    bos_min_px: int
    bos_max_px: int
    bos_y_px: int
    bos_min_cm: float
    bos_max_cm: float
    margin_of_stability_cm: float
    stability_index: float  # Normalized [-1.0 = unstable, 1.0 = centered]
    status: str             # "STABLE", "WARNING", or "UNSTABLE"
    posture_label: str
    segments: List[SegmentCOMResult]


def compute_base_of_support(
    aug: Dict[str, Tuple[float, float, float]], frame_w: int, frame_h: int
) -> Tuple[int, int, int, str]:
    """
    Determines active ground contact boundaries (BOS) from ankle, heel, and toe landmarks.
    Automatically detects single-leg stance when one foot is significantly elevated.
    """
    left_y = max(aug["left_heel"][1], aug["left_foot_index"][1], aug["left_ankle"][1])
    right_y = max(aug["right_heel"][1], aug["right_foot_index"][1], aug["right_ankle"][1])

    left_x_pts = [
        aug["left_ankle"][0] * frame_w,
        aug["left_heel"][0] * frame_w,
        aug["left_foot_index"][0] * frame_w,
    ]
    right_x_pts = [
        aug["right_ankle"][0] * frame_w,
        aug["right_heel"][0] * frame_w,
        aug["right_foot_index"][0] * frame_w,
    ]

    # Check if one foot is lifted off the ground
    if (right_y - left_y) > SINGLE_LEG_LIFT_THRESHOLD_RATIO:
        active_x = right_x_pts
        stance = "Single-Leg (Right)"
    elif (left_y - right_y) > SINGLE_LEG_LIFT_THRESHOLD_RATIO:
        active_x = left_x_pts
        stance = "Single-Leg (Left)"
    else:
        active_x = left_x_pts + right_x_pts
        hip_y = aug["mid_hip"][1]
        knee_y = 0.5 * (aug["left_knee"][1] + aug["right_knee"][1])
        if (knee_y - hip_y) < 0.11:
            stance = "Squatting"
        else:
            stance = "Bipedal Standing"

    pad_px = int(frame_w * 0.012)
    bos_min_px = int(round(min(active_x))) - pad_px
    bos_max_px = int(round(max(active_x))) + pad_px
    bos_y_px = int(round(max(left_y, right_y) * frame_h))

    return bos_min_px, bos_max_px, bos_y_px, stance


class BiomechanicalAnalyzer:
    """Computes human Center of Mass and postural stability from 2D pose landmarks."""

    def __init__(
        self,
        subject_height_cm: float = DEFAULT_SUBJECT_HEIGHT_CM,
        history_len: int = 90,
    ) -> None:
        self.subject_height_cm = subject_height_cm
        self.com_history_px: Deque[Tuple[int, int]] = deque(maxlen=history_len)
        self.velocity_window: Deque[float] = deque(maxlen=5)
        self.baseline_com_cm: Optional[Tuple[float, float]] = None
        self.calibrated_cm_per_px: Optional[float] = None
        self.prev_com_cm: Optional[Tuple[float, float]] = None
        self.prev_timestamp_s: Optional[float] = None

    def reset_baseline(self) -> None:
        """Resets the reference standing COM position and metric scale factor."""
        self.baseline_com_cm = None
        self.calibrated_cm_per_px = None
        self.velocity_window.clear()

    @staticmethod
    def _augment_virtual_joints(
        landmarks: Dict[str, Tuple[float, float, float]]
    ) -> Dict[str, Tuple[float, float, float]]:
        """Adds mid-shoulder and mid-hip virtual landmarks required by Winter's model."""
        aug = dict(landmarks)
        ls, rs = aug["left_shoulder"], aug["right_shoulder"]
        lh, rh = aug["left_hip"], aug["right_hip"]

        aug["mid_shoulder"] = (
            0.5 * (ls[0] + rs[0]),
            0.5 * (ls[1] + rs[1]),
            min(ls[2], rs[2]),
        )
        aug["mid_hip"] = (
            0.5 * (lh[0] + rh[0]),
            0.5 * (lh[1] + rh[1]),
            min(lh[2], rh[2]),
        )
        return aug

    def _get_cm_per_px_scale(
        self, aug_landmarks: Dict[str, Tuple[float, float, float]], frame_h: int
    ) -> float:
        """
        Computes and locks the centimeter-per-pixel scale factor from initial standing posture
        so that squatting does not artificially alter metric scaling.
        """
        nose_y = aug_landmarks["nose"][1] * frame_h
        ankle_y = max(
            aug_landmarks["left_heel"][1],
            aug_landmarks["right_heel"][1],
            aug_landmarks["left_ankle"][1],
            aug_landmarks["right_ankle"][1],
        ) * frame_h
        apparent_height_px = max(abs(ankle_y - nose_y) * 1.08, frame_h * 0.35)
        instant_scale = self.subject_height_cm / apparent_height_px

        if self.calibrated_cm_per_px is None:
            self.calibrated_cm_per_px = instant_scale
        else:
            # Update only if the subject stands taller than initial calibration
            self.calibrated_cm_per_px = min(self.calibrated_cm_per_px, instant_scale)

        return self.calibrated_cm_per_px

    @staticmethod
    def _compute_base_of_support(
        aug: Dict[str, Tuple[float, float, float]], frame_w: int, frame_h: int
    ) -> Tuple[int, int, int, str]:
        """Wrapper around top-level compute_base_of_support for class-level calls."""
        return compute_base_of_support(aug, frame_w, frame_h)

    def analyze_frame(
        self,
        landmarks: Dict[str, Tuple[float, float, float]],
        frame_w: int,
        frame_h: int,
        timestamp_s: float,
        frame_idx: int,
        posture_hint: str = "Dynamic",
    ) -> PosturalTelemetry:
        """
        Calculates segment COMs, total body COM, Base of Support, and Margin of Stability.

        Equation:
            COM = sum(m_i * r_i) / sum(m_i)
        """
        aug = self._augment_virtual_joints(landmarks)
        cm_per_px = self._get_cm_per_px_scale(aug, frame_h)

        segments_out: List[SegmentCOMResult] = []
        weighted_x = 0.0
        weighted_y = 0.0
        total_mass = 0.0

        for seg_def in ANTHROPOMETRIC_SEGMENTS:
            prox = aug[seg_def.proximal_joint]
            dist = aug[seg_def.distal_joint]

            com_x_norm = prox[0] + seg_def.com_ratio * (dist[0] - prox[0])
            com_y_norm = prox[1] + seg_def.com_ratio * (dist[1] - prox[1])

            prox_px = (int(round(prox[0] * frame_w)), int(round(prox[1] * frame_h)))
            dist_px = (int(round(dist[0] * frame_w)), int(round(dist[1] * frame_h)))
            com_px = (int(round(com_x_norm * frame_w)), int(round(com_y_norm * frame_h)))

            segments_out.append(
                SegmentCOMResult(
                    name=seg_def.name,
                    proximal_px=prox_px,
                    distal_px=distal_px if "distal_px" in locals() else dist_px,
                    com_px=com_px,
                    com_norm=(com_x_norm, com_y_norm),
                    mass_fraction=seg_def.mass_fraction,
                )
            )

            weighted_x += seg_def.mass_fraction * com_x_norm
            weighted_y += seg_def.mass_fraction * com_y_norm
            total_mass += seg_def.mass_fraction

        total_com_norm = (weighted_x / total_mass, weighted_y / total_mass)

        # Use continuous sub-pixel floating-point coordinates for accurate kinematics
        exact_com_x_px = total_com_norm[0] * frame_w
        exact_com_y_px = total_com_norm[1] * frame_h

        total_com_px = (int(round(exact_com_x_px)), int(round(exact_com_y_px)))
        self.com_history_px.append(total_com_px)

        # Convert exact sub-pixel COM to metric coordinates (cm)
        com_cm = (
            (exact_com_x_px - frame_w * 0.5) * cm_per_px,
            (frame_h - exact_com_y_px) * cm_per_px,
        )

        if self.baseline_com_cm is None:
            self.baseline_com_cm = com_cm

        com_displacement_cm = float(
            np.hypot(com_cm[0] - self.baseline_com_cm[0], com_cm[1] - self.baseline_com_cm[1])
        )

        # Calculate smoothed instantaneous sway velocity (cm/s)
        if self.prev_com_cm is not None and self.prev_timestamp_s is not None:
            dt = max(timestamp_s - self.prev_timestamp_s, 1e-3)
            dist_step = float(
                np.hypot(com_cm[0] - self.prev_com_cm[0], com_cm[1] - self.prev_com_cm[1])
            )
            raw_vel = dist_step / dt
            self.velocity_window.append(raw_vel)
            sway_velocity_cm_s = float(np.mean(self.velocity_window))
        else:
            sway_velocity_cm_s = 0.0

        self.prev_com_cm = com_cm
        self.prev_timestamp_s = timestamp_s

        # Compute Base of Support (BOS) active contact points
        bos_min_px, bos_max_px, bos_y_px, detected_stance = compute_base_of_support(
            aug, frame_w, frame_h
        )
        bos_min_cm = (bos_min_px - frame_w * 0.5) * cm_per_px
        bos_max_cm = (bos_max_px - frame_w * 0.5) * cm_per_px

        # Signed Margin of Stability (MoS): positive inside BOS, negative outside BOS
        dist_left_cm = com_cm[0] - bos_min_cm
        dist_right_cm = bos_max_cm - com_cm[0]
        margin_of_stability_cm = min(dist_left_cm, dist_right_cm)

        bos_width_cm = max(bos_max_cm - bos_min_cm, 1.0)
        half_bos_cm = 0.5 * bos_width_cm
        stability_index = float(np.clip(margin_of_stability_cm / half_bos_cm, -1.0, 1.0))

        # Classify stability state
        warning_threshold_cm = STABILITY_WARNING_MARGIN_RATIO * half_bos_cm
        if margin_of_stability_cm < 0.0:
            status = "UNSTABLE"
        elif margin_of_stability_cm < warning_threshold_cm:
            status = "WARNING"
        else:
            status = "STABLE"

        posture_label = posture_hint if posture_hint != "Dynamic" else detected_stance

        return PosturalTelemetry(
            timestamp_s=timestamp_s,
            frame_idx=frame_idx,
            com_px=total_com_px,
            com_norm=total_com_norm,
            com_cm=com_cm,
            com_displacement_cm=com_displacement_cm,
            sway_velocity_cm_s=sway_velocity_cm_s,
            bos_min_px=bos_min_px,
            bos_max_px=bos_max_px,
            bos_y_px=bos_y_px,
            bos_min_cm=bos_min_cm,
            bos_max_cm=bos_max_cm,
            margin_of_stability_cm=margin_of_stability_cm,
            stability_index=stability_index,
            status=status,
            posture_label=posture_label,
            segments=segments_out,
        )