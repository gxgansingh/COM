"""
Automated Matplotlib biomechanical report generator.
Reads recorded CSV session logs and outputs a 4-panel high-resolution figure
suitable for academic presentations and lab reports.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np


def load_telemetry_csv(csv_path: Path) -> Dict[str, np.ndarray]:
    """Reads the biomechanical telemetry CSV file into NumPy arrays."""
    columns: Dict[str, List[float]] = {
        "timestamp_s": [],
        "com_x_cm": [],
        "com_y_cm": [],
        "com_displacement_cm": [],
        "sway_velocity_cm_s": [],
        "bos_min_cm": [],
        "bos_max_cm": [],
        "margin_of_stability_cm": [],
        "stability_index": [],
    }
    statuses: List[str] = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in columns:
                columns[key].append(float(row[key]))
            statuses.append(row["status"])

    data = {k: np.array(v, dtype=np.float64) for k, v in columns.items()}
    data["status"] = np.array(statuses)
    return data


def generate_biomechanical_report(
    csv_path: Path,
    output_image_path: Path,
    show_plot: bool = False,
) -> Path:
    """
    Generates a 4-panel biomechanical stability analysis plot from CSV telemetry
    and prints summary statistics to the console.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Telemetry CSV file not found: {csv_path}")

    data = load_telemetry_csv(csv_path)
    t = data["timestamp_s"]
    if len(t) < 2:
        raise ValueError("Insufficient telemetry frames in CSV to generate a report.")

    com_x = data["com_x_cm"]
    com_y = data["com_y_cm"]
    bos_min = data["bos_min_cm"]
    bos_max = data["bos_max_cm"]
    mos = data["margin_of_stability_cm"]
    vel = data["sway_velocity_cm_s"]
    statuses = data["status"]

    # Compute summary biomechanical metrics
    step_distances = np.hypot(np.diff(com_x), np.diff(com_y))
    total_sway_path_cm = float(np.sum(step_distances))
    max_disp_cm = float(np.max(data["com_displacement_cm"]))
    min_mos_cm = float(np.min(mos))
    mean_vel_cm_s = float(np.mean(vel))
    stable_pct = float(np.mean(statuses == "STABLE") * 100.0)
    warning_pct = float(np.mean(statuses == "WARNING") * 100.0)
    unstable_pct = float(np.mean(statuses == "UNSTABLE") * 100.0)

    print("\n" + "=" * 68)
    print("BIOMECHANICAL POSTURAL STABILITY SUMMARY REPORT")
    print("=" * 68)
    print(f"Session Duration          : {t[-1] - t[0]:.2f} s ({len(t)} frames)")
    print(f"Total COM Sway Path Length: {total_sway_path_cm:.2f} cm")
    print(f"Peak COM Displacement     : {max_disp_cm:.2f} cm")
    print(f"Mean Postural Sway Speed  : {mean_vel_cm_s:.2f} cm/s")
    print(f"Minimum Margin (MoS)      : {min_mos_cm:+.2f} cm")
    print(
        f"Stability Distribution    : STABLE={stable_pct:.1f}% | "
        f"WARNING={warning_pct:.1f}% | UNSTABLE={unstable_pct:.1f}%"
    )
    print("=" * 68)

    # Configure Matplotlib figure aesthetics
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), dpi=200)
    fig.suptitle(
        "Vision-Based Center of Mass (COM) & Postural Stability Analysis Report\n"
        f"Total Sway Path: {total_sway_path_cm:.1f} cm | Peak Excursion: {max_disp_cm:.1f} cm | "
        f"Time Stable: {stable_pct:.1f}%",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: Lateral COM_X vs Dynamic Base of Support (BOS)
    ax1 = axes[0, 0]
    ax1.fill_between(
        t, bos_min, bos_max, color="#2ecc71", alpha=0.18, label="Base of Support (Safe Zone)"
    )
    ax1.plot(t, bos_min, color="#27ae60", linestyle="--", linewidth=1.2, label="BOS Left/Right Limits")
    ax1.plot(t, bos_max, color="#27ae60", linestyle="--", linewidth=1.2)
    ax1.plot(t, com_x, color="#e67e22", linewidth=2.2, label="Lateral COM ($COM_X$)")
    ax1.set_title("1. Lateral COM Trajectory vs. Base of Support", fontweight="bold")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Horizontal Position (cm)")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, alpha=0.35)

    # Panel 2: Vertical COM_Y Profile & Displacement
    ax2 = axes[0, 1]
    ax2.plot(t, com_y, color="#2980b9", linewidth=2.2, label="Vertical COM ($COM_Y$)")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        t,
        data["com_displacement_cm"],
        color="#8e44ad",
        linestyle=":",
        linewidth=1.8,
        label="Total Displacement from Baseline",
    )
    ax2.set_title("2. Vertical COM Profile & Total Excursion", fontweight="bold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Vertical COM Height (cm)", color="#2980b9")
    ax2_twin.set_ylabel("Total COM Displacement (cm)", color="#8e44ad")
    ax2.grid(True, alpha=0.35)

    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", frameon=True)

    # Panel 3: 2D Frontal-Plane Statokinesigram (COM_X vs COM_Y)
    ax3 = axes[1, 0]
    scatter = ax3.scatter(
        com_x, com_y, c=t, cmap="viridis", s=18, alpha=0.85, edgecolors="none", label="COM Path"
    )
    ax3.plot(com_x, com_y, color="#34495e", alpha=0.3, linewidth=1.0)
    ax3.scatter([com_x[0]], [com_y[0]], color="#2ecc71", s=90, marker="o", zorder=5, label="Start")
    ax3.scatter([com_x[-1]], [com_y[-1]], color="#e74c3c", s=90, marker="X", zorder=5, label="End")
    cbar = fig.colorbar(scatter, ax=ax3)
    cbar.set_label("Time (s)")
    ax3.set_title("3. Frontal Plane COM Trajectory (Statokinesigram)", fontweight="bold")
    ax3.set_xlabel("Mediolateral COM Position X (cm)")
    ax3.set_ylabel("Vertical COM Position Y (cm)")
    ax3.legend(loc="best", frameon=True)
    ax3.grid(True, alpha=0.35)

    # Panel 4: Margin of Stability (MoS) & Sway Velocity
    ax4 = axes[1, 1]
    ax4.axhline(0.0, color="#c0392b", linestyle="--", linewidth=1.5, label="BOS Boundary Threshold (MoS = 0)")
    ax4.fill_between(
        t, mos, 0.0, where=(mos < 0.0), color="#e74c3c", alpha=0.30, interpolate=True, label="Unstable Excursion"
    )
    ax4.plot(t, mos, color="#16a085", linewidth=2.2, label="Margin of Stability (MoS)")

    ax4_twin = ax4.twinx()
    ax4_twin.plot(
        t, vel, color="#d35400", alpha=0.55, linewidth=1.2, linestyle="-.", label="Sway Velocity (cm/s)"
    )
    ax4.set_title("4. Margin of Stability (MoS) & Postural Sway Velocity", fontweight="bold")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Margin of Stability (cm)", color="#16a085")
    ax4_twin.set_ylabel("Sway Velocity (cm/s)", color="#d35400")
    ax4.grid(True, alpha=0.35)

    lines_4a, labels_4a = ax4.get_legend_handles_labels()
    lines_4b, labels_4b = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines_4a + lines_4b, labels_4a + labels_4b, loc="lower left", frameon=True)

    plt.tight_layout(rect=[0, 0.01, 1, 0.94])
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_image_path, dpi=200)
    print(f"[INFO] Saved Matplotlib biomechanical report to: {output_image_path}")

    if show_plot:
        plt.show()
    plt.close(fig)
    return output_image_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a 4-panel Matplotlib biomechanical report from a recorded CSV log."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/com_telemetry.csv"),
        help="Path to input CSV telemetry log.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/biomechanical_report.png"),
        help="Path to output PNG/PDF report image.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display interactive Matplotlib window after saving.",
    )
    args = parser.parse_args()
    generate_biomechanical_report(args.csv, args.output, show_plot=args.show)


if __name__ == "__main__":
    main()