"""
Capture static hand-shape training samples from webcam.

Usage examples:
  # One-handed sign (captures when either hand is visible)
  python -m src.capture_dataset --label hello --hands one --samples 200

  # Two-handed sign (captures only when both hands are visible)
  python -m src.capture_dataset --label family --hands two --samples 200

  # Force a specific hand
  python -m src.capture_dataset --label thankyou --hands left --samples 200
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp

from src.landmarks_extraction import mediapipe_detection
from src.sign_features import FEATURES_PER_HAND, build_frame_features, normalize_label

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

CSV_COLUMNS = [
    "label",
    "hands_required",
    "left_present",
    "right_present",
] + [f"f{i}" for i in range(FEATURES_PER_HAND * 2)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Sign label to record.")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples to capture.")
    parser.add_argument("--output", default="data/handshape_dataset.csv", help="CSV output path.")
    parser.add_argument(
        "--hands",
        choices=["one", "two", "left", "right"],
        default="one",
        help=(
            "one: capture when at least one hand is visible; "
            "two: capture only when both hands are visible; "
            "left/right: capture only that hand."
        ),
    )
    return parser.parse_args()


def should_capture(hands_required: str, left_ok: bool, right_ok: bool) -> bool:
    if hands_required == "one":
        return left_ok or right_ok
    if hands_required == "two":
        return left_ok and right_ok
    if hands_required == "left":
        return left_ok
    if hands_required == "right":
        return right_ok
    return False


def write_header_if_needed(writer: csv.writer, header_written: bool) -> bool:
    if header_written:
        return True
    writer.writerow(CSV_COLUMNS)
    return True


def main() -> None:
    args = parse_args()
    label = normalize_label(args.label)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    count = 0
    header_written = output_path.exists() and output_path.stat().st_size > 0
    if header_written:
        with output_path.open("r", encoding="utf-8", newline="") as existing:
            first_line = existing.readline().strip()
        if first_line != ",".join(CSV_COLUMNS):
            print(
                "[WARN] Existing dataset header does not match the dual-hand format.\n"
                "       New captures will be written to data/handshape_dataset_v2.csv instead."
            )
            output_path = Path("data/handshape_dataset_v2.csv")
            header_written = output_path.exists() and output_path.stat().st_size > 0

    with output_path.open("a", newline="", encoding="utf-8") as csv_f:
        writer = csv.writer(csv_f)
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened() and count < args.samples:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                image, results = mediapipe_detection(frame, holistic)
                face = results.face_landmarks

                vector, left_ok, right_ok, _, _ = build_frame_features(
                    results.left_hand_landmarks,
                    results.right_hand_landmarks,
                    face,
                    image.shape,
                )

                if should_capture(args.hands, left_ok, right_ok):
                    header_written = write_header_if_needed(writer, header_written)
                    row = [label, args.hands, int(left_ok), int(right_ok)] + vector.tolist()
                    writer.writerow(row)
                    count += 1

                for hand_landmarks in (results.left_hand_landmarks, results.right_hand_landmarks):
                    if hand_landmarks is None:
                        continue
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=2),
                    )

                ready = should_capture(args.hands, left_ok, right_ok)
                status_color = (80, 220, 80) if ready else (80, 80, 220)
                cv2.putText(
                    image,
                    f"{label} {count}/{args.samples}  hands={args.hands}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    f"left={'Y' if left_ok else 'N'}  right={'Y' if right_ok else 'N'}  "
                    f"{'CAPTURING' if ready else 'waiting...'}",
                    (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("Capture Dataset", image)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Captured {count} samples for '{label}' (hands={args.hands}) to {output_path}")


if __name__ == "__main__":
    main()
