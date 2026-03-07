"""
Action Unit extraction using MediaPipe Face Mesh.

Replaces the original OpenFace-based extraction so the pipeline can run
on any platform (Linux/macOS/Windows) without a compiled C++ binary.

Outputs a CSV with the same 17 AU intensity columns that OpenFace would
produce (AU01_r … AU45_r), so downstream code is unaffected.
"""

import os
import cv2
import numpy as np
import pandas as pd

# ── Landmark helpers ───────────────────────────────────────────────────────────
# MediaPipe Face Mesh indices used to approximate each AU.
_LEFT_EYE  = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def _ear(pts, indices):
    """Eye-aspect-ratio from 6-point eye landmarks."""
    v1 = np.linalg.norm(pts[indices[1]] - pts[indices[5]])
    v2 = np.linalg.norm(pts[indices[2]] - pts[indices[4]])
    h  = np.linalg.norm(pts[indices[0]] - pts[indices[3]])
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0


def _compute_aus(landmarks) -> dict:
    """
    Return approximate AU intensities (0-5 scale) from 468 face-mesh points.
    Keys match OpenFace column names (without leading space).
    """
    pts = np.array(landmarks)

    face_h = np.linalg.norm(pts[10] - pts[152])  # forehead→chin
    if face_h == 0:
        face_h = 1.0
    face_w = np.linalg.norm(pts[234] - pts[454])  # jaw width
    if face_w == 0:
        face_w = 1.0

    avg_ear = (_ear(pts, _LEFT_EYE) + _ear(pts, _RIGHT_EYE)) / 2.0

    # Lip geometry
    lip_w  = np.linalg.norm(pts[61] - pts[291])
    lip_h  = np.linalg.norm(pts[13] - pts[14])
    jaw_op = np.linalg.norm(pts[0] - pts[17])

    lip_corner_y = (pts[61][1] + pts[291][1]) / 2.0 - pts[13][1]

    aus = {
        # AU01 – Inner Brow Raiser
        "AU01_r": ((np.linalg.norm(pts[107] - pts[159])
                   + np.linalg.norm(pts[336] - pts[386])) / 2) / face_h * 20,
        # AU02 – Outer Brow Raiser
        "AU02_r": ((np.linalg.norm(pts[70] - pts[33])
                   + np.linalg.norm(pts[300] - pts[362])) / 2) / face_h * 20,
        # AU04 – Brow Lowerer
        "AU04_r": max(0, (1 - np.linalg.norm(pts[107] - pts[336]) / face_w * 3)) * 5,
        # AU05 – Upper Lid Raiser
        "AU05_r": max(0, (avg_ear - 0.2)) * 10,
        # AU06 – Cheek Raiser
        "AU06_r": max(0, -((pts[145][1] - pts[117][1])
                          + (pts[374][1] - pts[346][1])) / 2 / face_h * 20),
        # AU07 – Lid Tightener
        "AU07_r": max(0, (0.3 - avg_ear)) * 10,
        # AU09 – Nose Wrinkler
        "AU09_r": max(0, (1 - np.linalg.norm(pts[6] - pts[1]) / face_h * 5)) * 3,
        # AU10 – Upper Lip Raiser
        "AU10_r": max(0, (1 - np.linalg.norm(pts[1] - pts[13]) / face_h * 5)) * 3,
        # AU12 – Lip Corner Puller (smile)
        "AU12_r": max(0, -lip_corner_y / face_h * 15),
        # AU14 – Dimpler
        "AU14_r": max(0, (lip_w / face_w - 0.4)) * 5,
        # AU15 – Lip Corner Depressor
        "AU15_r": max(0, lip_corner_y / face_h * 15),
        # AU17 – Chin Raiser
        "AU17_r": max(0, (1 - np.linalg.norm(pts[152] - pts[17]) / face_h * 4)) * 3,
        # AU20 – Lip Stretcher
        "AU20_r": max(0, (lip_w / face_w - 0.35)) * 5,
        # AU23 – Lip Tightener
        "AU23_r": max(0, (1 - lip_h / face_h * 8)) * 3,
        # AU25 – Lips Part
        "AU25_r": max(0, lip_h / face_h * 15),
        # AU26 – Jaw Drop
        "AU26_r": jaw_op / face_h * 15,
        # AU45 – Blink
        "AU45_r": max(0, (0.25 - avg_ear)) * 15,
    }

    # Clamp all values to [0, 5]
    return {k: max(0.0, min(5.0, v)) for k, v in aus.items()}


# ── Public API (same signature as before) ──────────────────────────────────────
def extract_aus(video_path: str, output_dir: str) -> None:
    """
    Extract per-frame AU intensities from *video_path* and write a CSV
    into *output_dir*.  Drop-in replacement for the old OpenFace wrapper.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    all_rows: list[dict] = []
    frame_num = 0

    import mediapipe as mp
    try:
        _mp_face_mesh = mp.solutions.face_mesh
    except AttributeError:
        from mediapipe.python.solutions import face_mesh as _mp_face_mesh

    with _mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mesh.process(rgb)

            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                h, w, _ = frame.shape
                landmarks = [(lm.x * w, lm.y * h, lm.z * w)
                             for lm in face.landmark]
                row = _compute_aus(landmarks)
                row["frame"] = frame_num
                all_rows.append(row)

            frame_num += 1

    cap.release()

    if not all_rows:
        raise RuntimeError("No faces detected in the video.")

    df = pd.DataFrame(all_rows)

    # Column order matching OpenFace output
    au_order = [
        "frame",
        "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
        "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
        "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
    ]
    for col in au_order:
        if col not in df.columns:
            df[col] = 0.0

    df = df[au_order]

    stem = os.path.splitext(os.path.basename(video_path))[0]
    df.to_csv(os.path.join(output_dir, f"{stem}.csv"), index=False)