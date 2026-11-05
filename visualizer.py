"""
Real-time OpenCV visualization engine for biomechanical skeletal overlays,
Center of Mass trajectory, Base of Support region, and telemetry HUD.
"""

from collections import deque
import math
from typing import Deque, Tuple
import cv2
import numpy as np

from biomechanics import PosturalTelemetry
from config import COLORS_BGR


class BiomechanicalVisualizer:
    """Renders biomechanical overlays and real-time stability telemetry onto video frames."""

    def __init__(self, strip_chart_len: int = 120) -> None:
        self.com_x_series: Deque[float] = deque(maxlen=strip_chart_len)
        self.bos_min_series: Deque[float] = deque(maxlen=strip_chart_len)
        self.bos_max_series: Deque[float] = deque(maxlen=strip_chart_len)

    def render(
        self,
        frame: np.ndarray,
        telemetry: PosturalTelemetry,
        com_trail_px: Deque[Tuple[int, int]],
    ) -> np.ndarray:
        """Draws all biomechanical layers and returns the annotated BGR image."""
        out = frame.copy()
        status_color = self._get_status_color(telemetry.status)

        # 1. Draw segment links and individual segment COMs
        for seg in telemetry.segments:
            cv2.line(out, seg.proximal_px, seg.distal_px, COLORS_BGR["skeleton"], 2, cv2.LINE_AA)
            cv2.circle(out, seg.proximal_px, 4, COLORS_BGR["joint"], -1, cv2.LINE_AA)
            cv2.circle(out, seg.com_px, 4, COLORS_BGR["segment_com"], -1, cv2.LINE_AA)

        # 2. Draw Base of Support (BOS) bar and vertical boundaries
        bos_y = telemetry.bos_y_px
        bos_min = telemetry.bos_min_px
        bos_max = telemetry.bos_max_px

        overlay = out.copy()
        cv2.rectangle(
            overlay,
            (bos_min, bos_y - 16),
            (bos_max, bos_y + 10),
            status_color,
            -1,
        )
        cv2.addWeighted(overlay, 0.28, out, 0.72, 0, out)

        cv2.line(out, (bos_min, bos_y), (bos_max, bos_y), status_color, 4, cv2.LINE_AA)
        cv2.line(out, (bos_min, bos_y - 25), (bos_min, bos_y + 12), status_color, 2, cv2.LINE_AA)
        cv2.line(out, (bos_max, bos_y - 25), (bos_max, bos_y + 12), status_color, 2, cv2.LINE_AA)
        cv2.putText(
            out,
            "Base of Support (BOS)",
            (bos_min, bos_y + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            status_color,
            1,
            cv2.LINE_AA,
        )

        # 3. Draw vertical Line of Gravity (COM projection onto BOS plane)
        com_x, com_y = telemetry.com_px
        self._draw_dashed_line(out, (com_x, com_y), (com_x, bos_y), status_color, thickness=2, gap=10)
        cv2.circle(out, (com_x, bos_y), 7, status_color, -1, cv2.LINE_AA)

        # 4. Draw historical COM trajectory trail
        if len(com_trail_px) > 1:
            pts = np.array(com_trail_px, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [pts], isClosed=False, color=COLORS_BGR["trajectory"], thickness=2, lineType=cv2.LINE_AA)

        # 5. Draw Total Body COM target crosshair
        cv2.circle(out, (com_x, com_y), 12, COLORS_BGR["total_com"], 2, cv2.LINE_AA)
        cv2.circle(out, (com_x, com_y), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.line(out, (com_x - 18, com_y), (com_x + 18, com_y), COLORS_BGR["total_com"], 2, cv2.LINE_AA)
        cv2.line(out, (com_x, com_y - 18), (com_x, com_y + 18), COLORS_BGR["total_com"], 2, cv2.LINE_AA)
        cv2.putText(
            out,
            "TOTAL COM",
            (com_x + 16, com_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLORS_BGR["total_com"],
            2,
            cv2.LINE_AA,
        )

        # 6. Update and draw telemetry HUD + real-time strip chart
        self.com_x_series.append(telemetry.com_cm[0])
        self.bos_min_series.append(telemetry.bos_min_cm)
        self.bos_max_series.append(telemetry.bos_max_cm)

        self._draw_hud_panel(out, telemetry, status_color)
        self._draw_strip_chart(out, status_color)

        return out

    def _draw_hud_panel(
        self, frame: np.ndarray, telemetry: PosturalTelemetry, status_color: Tuple[int, int, int]
    ) -> None:
        """Draws the top-left biomechanical telemetry dashboard."""
        panel_x, panel_y = 20, 20
        panel_w, panel_h = 390, 260

        sub = frame[panel_y : panel_y + panel_h, panel_x : panel_x + panel_w]
        bg = np.full_like(sub, COLORS_BGR["hud_bg"])
        cv2.addWeighted(bg, 0.85, sub, 0.15, 0, sub)
        frame[panel_y : panel_y + panel_h, panel_x : panel_x + panel_w] = sub
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            COLORS_BGR["hud_border"],
            1,
        )

        cv2.putText(
            frame,
            "BIOMECHANICAL TELEMETRY (21MEO103T)",
            (panel_x + 15, panel_y + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            COLORS_BGR["joint"],
            2,
            cv2.LINE_AA,
        )

        rows = [
            f"Posture Phase   : {telemetry.posture_label}",
            f"Stability State : {telemetry.status}",
            f"COM Position    : X={telemetry.com_cm[0]:+5.1f} cm, Y={telemetry.com_cm[1]:5.1f} cm",
            f"COM Displacement: {telemetry.com_displacement_cm:5.2f} cm",
            f"Sway Velocity   : {telemetry.sway_velocity_cm_s:5.2f} cm/s",
            f"Margin (MoS)    : {telemetry.margin_of_stability_cm:+5.2f} cm",
            f"Stability Index : {telemetry.stability_index * 100:5.1f}%",
        ]

        y_cursor = panel_y + 60
        for idx, text in enumerate(rows):
            color = status_color if idx == 1 else COLORS_BGR["text_primary"]
            weight = 2 if idx == 1 else 1
            cv2.putText(
                frame,
                text,
                (panel_x + 15, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                weight,
                cv2.LINE_AA,
            )
            y_cursor += 27

        cv2.putText(
            frame,
            "Keys: [Q] Quit & Plot Report  [R] Reset Baseline",
            (panel_x + 15, panel_y + panel_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            COLORS_BGR["text_secondary"],
            1,
            cv2.LINE_AA,
        )

    def _draw_strip_chart(self, frame: np.ndarray, status_color: Tuple[int, int, int]) -> None:
        """Draws a live scrolling time-series chart of COM_X relative to BOS limits."""
        if len(self.com_x_series) < 2:
            return

        h, w = frame.shape[:2]
        chart_w, chart_h = 360, 140
        x0, y0 = w - chart_w - 20, 20

        sub = frame[y0 : y0 + chart_h, x0 : x0 + chart_w]
        bg = np.full_like(sub, COLORS_BGR["hud_bg"])
        cv2.addWeighted(bg, 0.85, sub, 0.15, 0, sub)
        frame[y0 : y0 + chart_h, x0 : x0 + chart_w] = sub
        cv2.rectangle(frame, (x0, y0), (x0 + chart_w, y0 + chart_h), COLORS_BGR["hud_border"], 1)

        cv2.putText(
            frame,
            "Live Lateral COM_X vs BOS Boundaries (cm)",
            (x0 + 12, y0 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLORS_BGR["text_primary"],
            1,
            cv2.LINE_AA,
        )

        y_min, y_max = -25.0, 25.0
        plot_top, plot_bottom = y0 + 34, y0 + chart_h - 12
        plot_h = plot_bottom - plot_top

        def to_py(val_cm: float) -> int:
            norm = (np.clip(val_cm, y_min, y_max) - y_min) / (y_max - y_min)
            return int(plot_bottom - norm * plot_h)

        n_pts = len(self.com_x_series)
        max_pts = max((self.com_x_series.maxlen or 120) - 1, 1)
        for i in range(1, n_pts):
            px1 = x0 + 10 + int((i - 1) / max_pts * (chart_w - 20))
            px2 = x0 + 10 + int(i / max_pts * (chart_w - 20))

            # Draw BOS bounds in muted green
            cv2.line(
                frame,
                (px1, to_py(self.bos_min_series[i - 1])),
                (px2, to_py(self.bos_min_series[i])),
                (70, 140, 70),
                1,
                cv2.LINE_AA,
            )
            cv2.line(
                frame,
                (px1, to_py(self.bos_max_series[i - 1])),
                (px2, to_py(self.bos_max_series[i])),
                (70, 140, 70),
                1,
                cv2.LINE_AA,
            )
            # Draw COM_X trace
            cv2.line(
                frame,
                (px1, to_py(self.com_x_series[i - 1])),
                (px2, to_py(self.com_x_series[i])),
                status_color,
                2,
                cv2.LINE_AA,
            )

    @staticmethod
    def _get_status_color(status: str) -> Tuple[int, int, int]:
        if status == "STABLE":
            return COLORS_BGR["stable"]
        if status == "WARNING":
            return COLORS_BGR["warning"]
        return COLORS_BGR["unstable"]

    @staticmethod
    def _draw_dashed_line(
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 2,
        gap: int = 10,
    ) -> None:
        dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        if dist <= 0:
            return
        pts = []
        for i in np.arange(0.0, dist, gap):
            r = i / dist
            x = int(pt1[0] * (1.0 - r) + pt2[0] * r)
            y = int(pt1[1] * (1.0 - r) + pt2[1] * r)
            pts.append((x, y))
        for i in range(0, len(pts) - 1, 2):
            cv2.line(img, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)