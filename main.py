"""
Main execution entry point for Vision-Based Center of Mass Estimation
and Postural Stability Analysis (Course: 21MEO103T - Fundamentals of Biomechanics).
"""

import argparse
import csv
import time
from pathlib import Path
import cv2

from biomechanics import BiomechanicalAnalyzer, PosturalTelemetry
from config import DEFAULT_SUBJECT_HEIGHT_CM
from pose_detector import PoseDetector, SyntheticBiomechanicsSimulator
from report_generator import generate_biomechanical_report
from visualizer import BiomechanicalVisualizer


CSV_FIELDNAMES = [
    "frame_idx",
    "timestamp_s",
    "posture_label",
    "status",
    "com_x_px",
    "com_y_px",
    "com_x_cm",
    "com_y_cm",
    "com_displacement_cm",
    "sway_velocity_cm_s",
    "bos_min_cm",
    "bos_max_cm",
    "margin_of_stability_cm",
    "stability_index",
]


def write_telemetry_row(writer: csv.DictWriter, telemetry: PosturalTelemetry) -> None:
    """Writes a single frame's biomechanical metrics to the CSV log."""
    writer.writerow(
        {
            "frame_idx": telemetry.frame_idx,
            "timestamp_s": f"{telemetry.timestamp_s:.4f}",
            "posture_label": telemetry.posture_label,
            "status": telemetry.status,
            "com_x_px": telemetry.com_px[0],
            "com_y_px": telemetry.com_px[1],
            "com_x_cm": f"{telemetry.com_cm[0]:.3f}",
            "com_y_cm": f"{telemetry.com_cm[1]:.3f}",
            "com_displacement_cm": f"{telemetry.com_displacement_cm:.3f}",
            "sway_velocity_cm_s": f"{telemetry.sway_velocity_cm_s:.3f}",
            "bos_min_cm": f"{telemetry.bos_min_cm:.3f}",
            "bos_max_cm": f"{telemetry.bos_max_cm:.3f}",
            "margin_of_stability_cm": f"{telemetry.margin_of_stability_cm:.3f}",
            "stability_index": f"{telemetry.stability_index:.4f}",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vision-Based Center of Mass (COM) & Postural Stability Analyzer"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera index (e.g., '0') or path to an input video file.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run built-in synthetic biomechanical simulation (no webcam required).",
    )
    parser.add_argument(
        "--height-cm",
        type=float,
        default=DEFAULT_SUBJECT_HEIGHT_CM,
        help="Subject height in centimeters for metric calibration (default: 172.0).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Optional max session duration in seconds (0 = run until 'Q' is pressed).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening an OpenCV GUI window (useful for batch processing).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to save CSV telemetry and Matplotlib report.",
    )
    parser.add_argument(
        "--show-report",
        action="store_true",
        help="Open interactive Matplotlib report window when session finishes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "com_telemetry.csv"
    report_path = args.output_dir / "biomechanical_report.png"

    analyzer = BiomechanicalAnalyzer(subject_height_cm=args.height_cm)
    visualizer = BiomechanicalVisualizer()
    detector = PoseDetector()

    use_demo = args.demo
    cap = None

    if not use_demo:
        source = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("[WARNING] Could not open video source. Falling back to --demo simulation mode.")
            use_demo = True

    print("[INFO] Starting Biomechanical COM & Postural Stability Analyzer...")
    print(f"[INFO] Mode: {'Synthetic Biomechanical Simulation (--demo)' if use_demo else f'Live/Video ({args.source})'}")
    print(f"[INFO] Telemetry CSV Log: {csv_path}")

    start_time = time.time()
    frame_idx = 0

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        try:
            while True:
                elapsed_s = time.time() - start_time
                if args.duration > 0.0 and elapsed_s >= args.duration:
                    print(f"[INFO] Reached target session duration ({args.duration:.1f}s).")
                    break

                if use_demo:
                    # Use deterministic 30 FPS time step in headless batch runs
                    sim_t = (frame_idx / 30.0) if args.headless else elapsed_s
                    frame, landmarks, posture_hint = (
                        SyntheticBiomechanicsSimulator.generate_frame_and_landmarks(sim_t)
                    )
                    timestamp_s = sim_t
                else:
                    assert cap is not None
                    ret, frame = cap.read()
                    if not ret:
                        print("[INFO] End of video stream.")
                        break
                    landmarks = detector.extract_landmarks(frame)
                    posture_hint = "Dynamic"
                    timestamp_s = elapsed_s

                if landmarks is not None:
                    h, w = frame.shape[:2]
                    telemetry = analyzer.analyze_frame(
                        landmarks=landmarks,
                        frame_w=w,
                        frame_h=h,
                        timestamp_s=timestamp_s,
                        frame_idx=frame_idx,
                        posture_hint=posture_hint,
                    )
                    write_telemetry_row(writer, telemetry)
                    annotated_frame = visualizer.render(
                        frame, telemetry, analyzer.com_history_px
                    )
                else:
                    annotated_frame = frame.copy()
                    cv2.putText(
                        annotated_frame,
                        "No Human Pose Detected - Ensure Full Body (Head to Feet) is Visible",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                frame_idx += 1

                if not args.headless:
                    cv2.imshow(
                        "Vision-Based COM Estimation & Postural Stability (21MEO103T)",
                        annotated_frame,
                    )
                    key = cv2.waitKey(1 if not use_demo else 30) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        print("[INFO] User terminated session.")
                        break
                    if key in (ord("r"), ord("R")):
                        analyzer.reset_baseline()
                        print("[INFO] Baseline COM reference recalibrated.")
        finally:
            if cap is not None:
                cap.release()
            detector.close()
            if not args.headless:
                cv2.destroyAllWindows()

    # Automatically generate the 4-panel Matplotlib biomechanical report
    if frame_idx > 2:
        generate_biomechanical_report(
            csv_path=csv_path,
            output_image_path=report_path,
            show_plot=args.show_report,
        )


if __name__ == "__main__":
    main()