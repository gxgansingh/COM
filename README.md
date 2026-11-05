# Vision-Based Center of Mass (COM) Estimation & Postural Stability Analysis

**Domain:** Markerless Motion Capture · Postural Control & Balance

> *"From a simple camera feed to a measurable biomechanical representation of human balance."*

---

## 1. Project Overview & Motivation

Human postural stability and balance depend on the dynamic relationship between the body's **Center of Mass (COM)**, the vertical **Line of Gravity (LoG)**, and the **Base of Support (BOS)** formed by the feet in contact with the ground. During everyday movements, clinical rehabilitation exercises, and athletic tasks—such as quiet standing, lateral leaning, squatting, and single-leg stance—the COM shifts continuously in both horizontal and vertical planes.

Traditionally, quantifying whole-body COM kinematics and postural sway requires expensive, laboratory-bound equipment such as **multi-camera optical motion capture systems (e.g., Vicon, OptiTrack)** and **piezoelectric/strain-gauge force platforms (e.g., AMTI, Bertec)**. While accurate, these systems are costly, non-portable, and inaccessible for routine classroom demonstrations, home-based physical therapy, or field sports coaching.

### What This Project Solves
This project implements a **low-cost, markerless, vision-based biomechanical analysis pipeline** in Python. Using a standard webcam, pre-recorded video, or a built-in synthetic kinematic simulator, the system:
1. Extracts 2D anatomical landmarks in real time using **MediaPipe Pose**.
2. Reconstructs a **14-segment biomechanical model** of the human body using standard anthropometric tables (**Winter, 2009; de Leva, 1996**).
3. Computes individual **segment COMs** and the **whole-body Total COM** with sub-pixel precision.
4. Dynamically tracks the active **Base of Support (BOS)**—automatically adapting to bipedal stance, squatting, and single-leg balance.
5. Quantifies real-time postural metrics: **Margin of Stability (MoS)**, **COM Displacement**, **Postural Sway Velocity**, and **Stability Index**.
6. Renders a live **Biomechanical Telemetry HUD** with an embedded time-series strip chart, exports frame-by-frame **CSV telemetry** and **annotated MP4 video**, and automatically generates a **4-panel publication-grade Matplotlib analytical report**[cite: 2].

---

## 2. System Pipeline & Architecture

The software follows a modular, sequential biomechanical processing pipeline:

```text
  [ Live Webcam / Video File / Synthetic Kinematic Simulator (--demo) ]
                                   │
                                   ▼
                     [ RGB Frame Preprocessing ]
                                   │
                                   ▼
            [ Human Pose Estimation (MediaPipe Pose 33-Keypoint) ]
                                   │
                                   ▼
       [ Anatomical Landmark Extraction + Virtual Joint Synthesis ]
             (Adds Mid-Shoulder & Mid-Hip Reference Centers)
                                   │
                                   ▼
         [ 14-Segment Anthropometric Modelling (Winter, 2009) ]
         (Proximal-to-Distal Interpolation + Segment Mass Weighting)
                                   │
                                   ▼
           [ Sub-Pixel Total Body COM & Metric Calibration ]
                 (Locked Standing Reference Scale in cm)
                                   │
                                   ▼
       [ Dynamic Base of Support (BOS) & Stance Classification ]
          (Bipedal Standing, Squatting, Left/Right Single-Leg)
                                   │
                                   ▼
        [ Postural Stability & Sway Kinematics Computation ]
      (Margin of Stability, Sway Velocity, Excursion, Status State)
                                   │
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                      ▼
  [ Live OpenCV HUD ]    [ CSV Telemetry Log ]   [ 4-Panel Matplotlib ]
  (& Annotated .MP4)     (com_telemetry.csv)     (Analytical Report)
```

---

## 3. Biomechanical & Mathematical Foundation

### 3.1 Virtual Joint Synthesis
Standard pose estimation detectors provide bilateral shoulder and hip keypoints (`left_shoulder`, `right_shoulder`, `left_hip`, `right_hip`), whereas Winter's axial segment definitions require central trunk endpoints. Prior to segment evaluation, the engine synthesizes two virtual joints:

$$\mathbf{p}_{\text{mid\_shoulder}} = \frac{\mathbf{p}_{\text{left\_shoulder}} + \mathbf{p}_{\text{right\_shoulder}}}{2}, \quad \mathbf{p}_{\text{mid\_hip}} = \frac{\mathbf{p}_{\text{left\_hip}} + \mathbf{p}_{\text{right\_hip}}}{2}$$

### 3.2 Individual Segment COM Calculation
For each of the $N = 14$ anatomical segments, the segment Center of Mass $\mathbf{r}_i = (x_i, y_i)$ lies along the vector connecting its **proximal joint** $\mathbf{p}_{\text{prox}, i}$ to its **distal joint** $\mathbf{p}_{\text{dist}, i}$ at an anthropometric fraction $c_i$:

$$\mathbf{r}_i = \mathbf{p}_{\text{prox}, i} + c_i \left( \mathbf{p}_{\text{dist}, i} - \mathbf{p}_{\text{prox}, i} \right)$$

### 3.3 Whole-Body Center of Mass (Total COM)
By applying the principle of moments (Varignon's Theorem), the total body Center of Mass $\mathbf{R}_{\text{COM}} = (X_{\text{COM}}, Y_{\text{COM}})$ is the mass-weighted average of all 14 segment COMs:

$$\mathbf{R}_{\text{COM}} = \frac{\sum_{i=1}^{14} m_i \, \mathbf{r}_i}{\sum_{i=1}^{14} m_i}$$

where $m_i$ is the segment mass fraction ($m_i / M_{\text{total}}$) and $\sum_{i=1}^{14} m_i = 1.000$.

### 3.4 14-Segment Anthropometric Model Table (Winter, 2009)
The exact anthropometric constants implemented in `config.py` are listed below:

| # | Segment Name | Proximal Landmark | Distal Landmark | Mass Fraction ($m_i$) | COM Ratio from Proximal ($c_i$) |
| :- | :--- | :--- | :--- | :---: | :---: |
| 1 | **Head & Neck** | `mid_shoulder` | `nose` | `0.0810` | `0.500` |
| 2 | **Trunk (Thorax + Abdomen + Pelvis)** | `mid_shoulder` | `mid_hip` | `0.4970` | `0.500` |
| 3 | **Left Upper Arm** | `left_shoulder` | `left_elbow` | `0.0280` | `0.436` |
| 4 | **Right Upper Arm** | `right_shoulder` | `right_elbow` | `0.0280` | `0.436` |
| 5 | **Left Forearm** | `left_elbow` | `left_wrist` | `0.0160` | `0.430` |
| 6 | **Right Forearm** | `right_elbow` | `right_wrist` | `0.0160` | `0.430` |
| 7 | **Left Hand** | `left_wrist` | `left_wrist` | `0.0060` | `0.506` |
| 8 | **Right Hand** | `right_wrist` | `right_wrist` | `0.0060` | `0.506` |
| 9 | **Left Thigh** | `left_hip` | `left_knee` | `0.1000` | `0.433` |
| 10 | **Right Thigh** | `right_hip` | `right_knee` | `0.1000` | `0.433` |
| 11 | **Left Shank (Lower Leg)** | `left_knee` | `left_ankle` | `0.0465` | `0.433` |
| 12 | **Right Shank (Lower Leg)** | `right_knee` | `right_ankle` | `0.0465` | `0.433` |
| 13 | **Left Foot** | `left_heel` | `left_foot_index` | `0.0145` | `0.500` |
| 14 | **Right Foot** | `right_heel` | `right_foot_index` | `0.0145` | `0.500` |
| **—** | **Total Body** | — | — | **`1.0000` (100%)** | — |

### 3.5 Sub-Pixel Metric Calibration & Standing Scale Lock
To convert pixel coordinates into physical centimeters ($\text{cm}$) without requiring a physical checkerboard in the scene:
1. The subject's known height $H_{\text{cm}}$ (configurable via `--height-cm`, default `172.0 cm`) is divided by the apparent vertical pixel span from the lowest ground-contact foot landmark to the cranial vertex (`1.08 × |y_ankle - y_nose|`).
2. **Squat Distortion Prevention:** If the pixel-to-centimeter scale were recalculated naively on every frame, lowering the head during a squat would make the algorithm assume the person shrank, artificially inflating metric distances. To prevent this, `biomechanics.py` locks `calibrated_cm_per_px` to the tallest standing posture observed (or resets upon pressing **`R`**).
3. **Sub-Pixel Kinematics:** All metric conversions ($\text{COM}_X$, $\text{COM}_Y$, displacement, and velocity) are computed directly from floating-point normalized coordinates prior to integer pixel rounding, eliminating quantization stair-stepping and velocity spikes.

### 3.6 Dynamic Base of Support (BOS) & Single-Leg Detection
The Base of Support in the frontal/coronal plane is bounded horizontally by $\left[\text{BOS}_{\min}, \text{BOS}_{\max}\right]$:
* **Bipedal Stance:** Uses the horizontal span across all six foot landmarks (`left/right ankle`, `heel`, and `foot_index`).
* **Single-Leg Stance Detection:** If the vertical elevation difference between the left and right feet exceeds `SINGLE_LEG_LIFT_THRESHOLD_RATIO` ($4.5\%$ of frame height), the lifted foot is excluded from ground contact, and the active BOS automatically contracts to the supporting foot alone.
* **Squat Classification:** When both feet remain grounded and the normalized hip-to-knee vertical separation drops below `0.11`, the posture classifier labels the phase as `Squatting`.

### 3.7 Margin of Stability (MoS), Stability Index & State Classification
* **Signed Margin of Stability ($\text{MoS}$):** Measures the minimum horizontal distance from the vertical projection of the COM (Line of Gravity) to the nearest active BOS boundary:
  $$\text{MoS} = \min\left( X_{\text{COM}} - \text{BOS}_{\min}, \; \text{BOS}_{\max} - X_{\text{COM}} \right)$$
  * $\text{MoS} > 0\text{ cm}$: Line of Gravity is **inside** the Base of Support[cite: 2].
  * $\text{MoS} = 0\text{ cm}$: Line of Gravity is **on the edge** of the Base of Support[cite: 2].
  * $\text{MoS} < 0\text{ cm}$: Line of Gravity has **exited** the Base of Support (Unstable / Fall Risk)[cite: 2].

* **Normalized Stability Index ($SI$):**
  $$SI = \text{clip}\left( \frac{\text{MoS}}{0.5 \times (\text{BOS}_{\max} - \text{BOS}_{\min})}, \; -1.0, \; +1.0 \right)$$
  where $100\%$ indicates the COM is dead-center over the Base of Support, $0\%$ is on the boundary edge, and negative values indicate boundary excursion.

* **Three-Tier Stability Classification:**
  * 🟢 **`STABLE`**: $\text{MoS} \ge 0.20 \times \left(\frac{\text{BOS}_{\text{width}}}{2}\right)$
  * 🟡 **`WARNING`**: $0 \le \text{MoS} < 0.20 \times \left(\frac{\text{BOS}_{\text{width}}}{2}\right)$ (within the outer $20\%$ warning buffer of the BOS)
  * 🔴 **`UNSTABLE`**: $\text{MoS} < 0\text{ cm}$ (Line of Gravity outside the BOS)

### 3.8 Postural Sway Kinematics
* **COM Excursion / Displacement ($D_{\text{COM}}$):** Euclidean distance from the initial calibrated quiet-standing baseline COM $(X_0, Y_0)$:
  $$D_{\text{COM}}(t) = \sqrt{\left(X_{\text{COM}}(t) - X_0\right)^2 + \left(Y_{\text{COM}}(t) - Y_0\right)^2}$$
* **Instantaneous Sway Velocity ($V_{\text{sway}}$):** Filtered using a 5-frame moving average window to suppress high-frequency camera jitter:
  $$V_{\text{sway}}(t_k) = \frac{1}{W} \sum_{j=0}^{W-1} \frac{\Vert{}\mathbf{R}_{\text{COM}}(t_{k-j}) - \mathbf{R}_{\text{COM}}(t_{k-j-1})\Vert{}_2}{\Delta t_{k-j}}$$
* **Total Sway Path Length ($L_{\text{path}}$):** Cumulative trajectory length traveled by the COM over an entire session of $K$ frames:
  $$L_{\text{path}} = \sum_{k=1}^{K-1} \sqrt{\left(X_{\text{COM}}(t_k) - X_{\text{COM}}(t_{k-1})\right)^2 + \left(Y_{\text{COM}}(t_k) - Y_{\text{COM}}(t_{k-1})\right)^2}$$

---

## 4. Repository Structure & Module Breakdown

```text
COM/
├── .gitignore                 # Git ignore rules for venv, cache, and generated outputs
├── requirements.txt           # Python package dependencies
├── README.md                  # Comprehensive project documentation
├── config.py                  # Winter (2009) 14-segment constants, landmarks & BGR palette
├── biomechanics.py            # Core COM, BOS, MoS, metric scaling & velocity algorithms
├── pose_detector.py           # MediaPipe Pose wrapper & 5-phase synthetic kinematic simulator
├── visualizer.py              # OpenCV HUD, skeleton, BOS bar & live strip-chart renderer
├── report_generator.py        # Automated 4-panel Matplotlib report & summary stats engine
├── main.py                    # CLI entry point, video capture/recording & CSV logger
└── outputs/                   # Auto-created directory for generated session artifacts
    ├── com_telemetry.csv      # Frame-by-frame biomechanical telemetry log
    ├── annotated_session.mp4  # Recorded HUD video (when --save-video is enabled)
    └── biomechanical_report.png # 4-panel high-resolution Matplotlib analytical report
```

### Detailed Module Responsibilities
* **`config.py`**: Defines the immutable `SegmentDefinition` dataclass, maps the 17 essential MediaPipe Pose landmark indices, stores the 14-segment `ANTHROPOMETRIC_SEGMENTS` tuple, sets biomechanical thresholds (`DEFAULT_SUBJECT_HEIGHT_CM`, `SINGLE_LEG_LIFT_THRESHOLD_RATIO`, `STABILITY_WARNING_MARGIN_RATIO`), and centralizes the OpenCV BGR color palette.
* **`biomechanics.py`**: Houses `SegmentCOMResult`, `PosturalTelemetry`, `compute_base_of_support()`, and `BiomechanicalAnalyzer`. Handles virtual joint synthesis, locked standing metric calibration, weighted COM summation, dynamic BOS contact detection, Margin of Stability computation, and smoothed sway velocity calculation.
* **`pose_detector.py`**: Provides `PoseDetector` (interfacing with `mediapipe.solutions.pose.Pose` with temporal landmark smoothing) and `SyntheticBiomechanicsSimulator`, which generates a deterministic 20-second, 5-phase biomechanical test loop when no camera is attached or `--demo` is passed.
* **`visualizer.py`**: Implements `BiomechanicalVisualizer`. Draws anatomical segment bones, joint nodes, cyan segment COM dots, the orange Total COM crosshair, a 90-frame trajectory trail, a dashed vertical Line of Gravity, a translucent Base of Support bar, a top-left telemetry dashboard, and a top-right scrolling strip chart of $COM_X$ vs. BOS limits.
* **`report_generator.py`**: Standalone and integrated post-processing module. Reads `outputs/com_telemetry.csv`, computes whole-session summary statistics in the terminal, and renders a non-overlapping, `250 DPI` 4-panel Matplotlib figure (`outputs/biomechanical_report.png`).
* **`main.py`**: Orchestrates CLI argument parsing, webcam/video/demo initialization, real-time frame processing, CSV streaming, optional `.mp4` video recording, keyboard event handling, and automatic report generation upon exit.

---

## 5. Installation & Environment Setup

### Prerequisites
* **Python:** `3.9` to `3.12` recommended (64-bit)
* **Operating System:** Windows 10/11, macOS, or Linux
* **Hardware:** Standard USB/integrated webcam (optional; `--demo` mode runs without hardware)

### Step 1: Clone or Navigate to the Project Directory
```powershell
cd D:\Projects\COM
```

### Step 2: Create a Virtual Environment
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment

* **Windows PowerShell:**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(Note: If PowerShell displays a `running scripts is disabled on this system` error, run the following command first to allow activation in the current session, then re-run `Activate.ps1`:)*
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
  ```

* **Windows Command Prompt (`cmd.exe`):**
  ```cmd
  venv\Scripts\activate.bat
  ```

* **macOS / Linux (`bash` / `zsh`):**
  ```bash
  source venv/bin/activate
  ```

### Step 4: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

---

## 6. Usage & Command-Line Interface (CLI)

### 6.1 Running `main.py` (Main Application)

| CLI Flag / Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--demo` | Flag | `False` | Runs the built-in 5-phase synthetic biomechanical simulation (no webcam needed). |
| `--source` | `str` | `"0"` | Camera device index (`"0"`, `"1"`) or path to an input video file (`"squat_test.mp4"`). |
| `--height-cm` | `float` | `172.0` | Subject standing height in centimeters for metric calibration. |
| `--duration` | `float` | `0.0` | Maximum session duration in seconds (`0.0` = run indefinitely until `Q` is pressed). |
| `--save-video` | Flag | `False` | Records the annotated OpenCV HUD session to `outputs/annotated_session.mp4`. |
| `--show-report` | Flag | `False` | Opens the interactive 4-panel Matplotlib report window automatically when finished. |
| `--headless` | Flag | `False` | Runs without opening an OpenCV GUI window (ideal for fast batch video processing). |
| `--output-dir` | `Path` | `outputs` | Target directory where CSV logs, videos, and PNG reports are saved. |

### 6.2 Common Execution Workflows

1. **Run the 20-Second Synthetic Simulation, Record MP4 Video, and View the Matplotlib Report:**
   ```powershell
   python main.py --demo --duration 20 --save-video --show-report
   ```

2. **Run Live Webcam Analysis (Calibrated for a 175 cm Subject):**
   ```powershell
   python main.py --source 0 --height-cm 175 --save-video --show-report
   ```
   *(Ensure you stand back far enough so your entire body—from head to feet—is visible in the camera frame.)*

3. **Analyze a Pre-Recorded Video File (e.g., Smartphone Video):**
   ```powershell
   python main.py --source path/to/movement_video.mp4 --height-cm 168 --save-video --show-report
   ```

4. **Fast Headless Batch Run (Generates CSV + PNG Report Without GUI Delay):**
   ```powershell
   python main.py --demo --duration 20 --headless
   ```

5. **Regenerate or Customize the Matplotlib Report Standalone from an Existing CSV:**
   ```powershell
   python report_generator.py --csv outputs/com_telemetry.csv --output outputs/biomechanical_report.png --show
   ```

### 6.3 Interactive Keyboard Controls (OpenCV Window)
While the real-time OpenCV visualization window is active:
* Press **`Q`** or **`ESC`**: Stop the session, close the video stream, save the CSV/MP4 files, and trigger the automated Matplotlib report generator.
* Press **`R`**: Recalibrate the subject's standing baseline COM reference position $(X_0, Y_0)$ and metric height scale factor.

---

## 7. Built-In 5-Phase Synthetic Biomechanical Simulation (`--demo`)

To enable reproducible testing, classroom demonstrations, and verification of edge cases without needing a live test subject, `SyntheticBiomechanicsSimulator` in `pose_detector.py` executes a continuous **20-second biomechanical test protocol** divided into five 4-second stages:

| Time Window | Posture Phase Label | Kinematic Behavior | Expected Stability Outcome |
| :---: | :--- | :--- | :--- |
| **`0.0 – 4.0 s`** | `Quiet Standing` | Upright bipedal posture with realistic physiological micro-sway ($0.5\text{ Hz}$). | 🟢 `STABLE` ($\text{MoS} \approx +33\text{ cm}$)[cite: 2] |
| **`4.0 – 8.0 s`** | `Lateral Lean (Within BOS)` | Controlled mediolateral trunk and hip shift to the right (+X) up to $+18\text{ cm}$[cite: 2]. | 🟢 `STABLE` ($\text{MoS}$ decreases to $\approx +16\text{ cm}$)[cite: 2] |
| **`8.0 – 12.0 s`** | `Deep Squatting` | Bilateral knee and hip flexion lowering vertical $\text{COM}_Y$ from $128.5\text{ cm}$ to $103.0\text{ cm}$[cite: 2]. | 🟢 `STABLE` (BOS stays constant; vertical excursion peaks at $\approx 26\text{ cm}$)[cite: 2] |
| **`12.0 – 16.0 s`** | `Single-Leg Balance` | Left foot lifts off the ground; pelvis and torso shift laterally over the right stance foot ($-27\text{ cm}$)[cite: 2]. | 🟢/🟡 `STABLE` $\rightarrow$ `WARNING` (BOS contracts sharply to the right foot only)[cite: 2] |
| **`16.0 – 20.0 s`** | `Extreme Lean (Unstable)` | Aggressive lateral lean (+X up to $+44.3\text{ cm}$) pushing the Line of Gravity outside the right foot[cite: 2]. | 🔴 `UNSTABLE` ($\text{MoS}$ drops to $-10\text{ cm}$, triggering red excursion alert)[cite: 2] |

---

## 8. Outputs & Analytical Deliverables

Every session automatically populates the `outputs/` directory with up to three deliverables:

### 8.1 Real-Time OpenCV HUD & Annotated Video (`outputs/annotated_session.mp4`)
* **Skeletal Overlay:** Grey anatomical segment links with orange joint landmarks and cyan segment COM markers ($\bullet$).
* **Total Body COM Marker:** High-visibility orange crosshair with a 90-frame trajectory trail.
* **Line of Gravity & BOS Bar:** Dashed vertical projection line intersecting a dynamically colored Base of Support bar (`Green = STABLE`, `Yellow = WARNING`, `Red = UNSTABLE`).
* **Top-Left Telemetry Panel:** Displays real-time Posture Phase, Stability State, $\text{COM}_X / \text{COM}_Y$ ($\text{cm}$), COM Displacement ($\text{cm}$), Sway Velocity ($\text{cm/s}$), Margin of Stability ($\text{cm}$), and Stability Index ($\%$).
* **Top-Right Live Strip Chart:** Scrolling 120-frame time series plotting $\text{COM}_X$ against the upper and lower BOS boundaries.

### 8.2 Telemetry CSV Log (`outputs/com_telemetry.csv`)
Contains frame-by-frame kinematic data formatted for direct import into MATLAB, Excel, R, or Python:

| Column Name | Unit | Description |
| :--- | :---: | :--- |
| `frame_idx` | `int` | Sequential zero-based frame counter. |
| `timestamp_s` | `s` | Elapsed session time in seconds. |
| `posture_label` | `str` | Active posture classification or simulation phase name. |
| `status` | `str` | Stability classification (`STABLE`, `WARNING`, `UNSTABLE`). |
| `com_x_px`, `com_y_px` | `px` | Integer image pixel coordinates of Total Body COM. |
| `com_x_cm`, `com_y_cm` | `cm` | Sub-pixel calibrated metric coordinates of Total Body COM. |
| `com_displacement_cm` | `cm` | Euclidean distance of COM from initial standing baseline. |
| `sway_velocity_cm_s` | `cm/s` | 5-frame smoothed instantaneous COM sway speed. |
| `bos_min_cm`, `bos_max_cm` | `cm` | Left and right horizontal boundaries of active Base of Support. |
| `margin_of_stability_cm` | `cm` | Signed distance from COM projection to nearest BOS edge ($\text{MoS}$). |
| `stability_index` | `[-1, 1]` | Normalized stability score relative to half-BOS width. |

### 8.3 Automated 4-Panel Matplotlib Report (`outputs/biomechanical_report.png`)
Generated automatically at `250 DPI` when a session completes[cite: 2]:
1. **Panel 1 — Lateral COM Trajectory vs. Base of Support:** Plots mediolateral $\text{COM}_X$ (orange) inside the shaded green Base of Support safe zone, clearly illustrating BOS narrowing during single-leg stance and boundary violation during extreme leaning[cite: 2].
2. **Panel 2 — Vertical COM Profile & Total Excursion:** Dual-axis plot showing vertical $\text{COM}_Y$ height changes (blue solid line, highlighting squat depth) alongside total Euclidean displacement from baseline (purple dashed line)[cite: 2].
3. **Panel 3 — Frontal Plane COM Trajectory (Statokinesigram):** 2D spatial scatter and path plot of $\text{COM}_X$ vs. $\text{COM}_Y$ color-coded by time (`viridis` colormap) with explicit `Start` (green circle) and `End` (red cross) markers[cite: 2].
4. **Panel 4 — Margin of Stability (MoS) & Postural Sway Velocity:** Dual-axis plot displaying $\text{MoS}$ (teal) relative to the $\text{MoS} = 0$ red dashed threshold, automatically shading unstable excursions ($\text{MoS} < 0$) in red alongside instantaneous sway velocity (orange dash-dot curve)[cite: 2].

---

## 9. Assumptions, Limitations & Future Scope

### Current Limitations
* **2D Planar Projection:** Single-camera analysis operates in the 2D image plane (primarily frontal/coronal or sagittal depending on camera placement) and does not capture simultaneous anteroposterior + mediolateral depth without a second camera or depth sensor.
* **Static vs. Dynamic Stability (XCoM):** The current stability criterion compares the vertical projection of the COM against the BOS. For high-velocity ballistic movements (e.g., sprinting or jumping), Hof's **Extrapolated Center of Mass ($\text{XCoM}$)** incorporating pendulum eigenfrequency $\omega_0 = \sqrt{g / l}$ can be added.
* **Anthropometric Generalization:** Cadaver-derived segment mass fractions (Winter, 2009) represent population averages and do not account for individual body composition variations or external loads (e.g., holding a barbell).

### Future Scope
* Integration of **Hof's Extrapolated Center of Mass ($\text{XCoM}$)** for dynamic gait and jump landing analysis.
* Addition of **joint angle kinematics** (knee flexion, hip flexion, and trunk lean angles in degrees).
* Development of an interactive **Streamlit Web Dashboard** for clinical trial comparison and PDF report export.
* Validation against laboratory **ground-reaction force plates** (Center of Pressure $\text{COP}$ vs. vision-estimated $\text{COM}$).

---

## 10. Academic References

1. **Winter, D. A.** (2009). *Biomechanics and Motor Control of Human Movement* (4th ed.). John Wiley & Sons. (Anthropometric segment mass fractions and center-of-mass ratios, Chapter 4).
2. **de Leva, P.** (1996). Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters. *Journal of Biomechanics*, 29(9), 1223–1230.
3. **Hof, A. L., Gazendam, M. G. J., & Sinke, W. E.** (2005). The condition for dynamic stability. *Journal of Biomechanics*, 38(1), 1–8.
4. **Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F., & Grundmann, M.** (2020). BlazePose: On-device real-time body pose tracking. *arXiv preprint arXiv:2006.10204*. (MediaPipe Pose architecture).