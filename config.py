"""
Configuration and anthropometric constants for Vision-Based Center of Mass (COM)
and Postural Stability Analysis.

Reference:
Winter, D. A. (2009). Biomechanics and Motor Control of Human Movement (4th ed.).
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class SegmentDefinition:
    """Defines a biomechanical body segment using anthropometric ratios."""
    name: str
    proximal_joint: str
    distal_joint: str
    mass_fraction: float  # Fraction of total body mass (m_i / M)
    com_ratio: float      # Distance of COM from proximal joint as a fraction of segment length


# MediaPipe Pose landmark name to index mapping
LANDMARK_INDICES: Dict[str, int] = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

# 14-Segment Anthropometric Model (Winter, 2009)
# Total mass fraction sums to exactly 1.000
ANTHROPOMETRIC_SEGMENTS: Tuple[SegmentDefinition, ...] = (
    SegmentDefinition("Head & Neck", "mid_shoulder", "nose", 0.0810, 0.500),
    SegmentDefinition("Trunk", "mid_shoulder", "mid_hip", 0.4970, 0.500),
    SegmentDefinition("Left Upper Arm", "left_shoulder", "left_elbow", 0.0280, 0.436),
    SegmentDefinition("Right Upper Arm", "right_shoulder", "right_elbow", 0.0280, 0.436),
    SegmentDefinition("Left Forearm", "left_elbow", "left_wrist", 0.0160, 0.430),
    SegmentDefinition("Right Forearm", "right_elbow", "right_wrist", 0.0160, 0.430),
    SegmentDefinition("Left Hand", "left_wrist", "left_wrist", 0.0060, 0.506),
    SegmentDefinition("Right Hand", "right_wrist", "right_wrist", 0.0060, 0.506),
    SegmentDefinition("Left Thigh", "left_hip", "left_knee", 0.1000, 0.433),
    SegmentDefinition("Right Thigh", "right_hip", "right_knee", 0.1000, 0.433),
    SegmentDefinition("Left Shank", "left_knee", "left_ankle", 0.0465, 0.433),
    SegmentDefinition("Right Shank", "right_knee", "right_ankle", 0.0465, 0.433),
    SegmentDefinition("Left Foot", "left_heel", "left_foot_index", 0.0145, 0.500),
    SegmentDefinition("Right Foot", "right_heel", "right_foot_index", 0.0145, 0.500),
)

# Default biomechanical calibration parameters
DEFAULT_SUBJECT_HEIGHT_CM: float = 172.0
SINGLE_LEG_LIFT_THRESHOLD_RATIO: float = 0.045  # Normalized vertical gap to detect foot lift
STABILITY_WARNING_MARGIN_RATIO: float = 0.20    # Inner 20% near BOS edge triggers warning

# OpenCV BGR Color Palette
COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "skeleton": (220, 220, 220),
    "joint": (255, 190, 0),
    "segment_com": (255, 255, 0),
    "total_com": (0, 140, 255),
    "trajectory": (255, 180, 50),
    "stable": (80, 220, 80),
    "warning": (0, 210, 255),
    "unstable": (60, 60, 255),
    "hud_bg": (24, 24, 28),
    "hud_border": (90, 90, 100),
    "text_primary": (245, 245, 245),
    "text_secondary": (180, 180, 190),
}