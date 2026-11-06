"""
Real-time static hand-shape sign detection.

This script focuses on classifying signs from hand shape (no motion tracking),
then draws a colored box around each detected hand and writes the sign label
below the box.
"""

import re
import subprocess
import threading
import time
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from src.landmarks_extraction import load_json_file, mediapipe_detection
from src.runtime_sign_classifier import RuntimeSignClassifier
from src.sign_features import hand_to_pixels

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

SIGNS_FILE = Path("signs.json")
MIN_PALM_SIZE_PX = 40
MODEL_FILE = Path("models/handshape_classifier.joblib")

PHRASE_CLEAR_TIMEOUT = 3.0
SIGN_CONFIRM_DURATION = 0.2
TTS_TRIGGER_OFFSET = 1.0


def speak(text: str) -> None:
    """Speak text using the OS built-in TTS engine on a daemon thread."""
    os_name = platform.system()

    def _run() -> None:
        if os_name == "Windows":
            safe = text.replace("'", "''")
            subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"Add-Type -AssemblyName System.Speech; "
                    f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f"$s.Speak('{safe}')",
                ],
                capture_output=True,
            )
        elif os_name == "Darwin":
            subprocess.run(["say", text], capture_output=True)
        else:
            if subprocess.run(["which", "espeak"], capture_output=True).returncode == 0:
                subprocess.run(["espeak", text], capture_output=True)
            else:
                subprocess.run(["spd-say", text], capture_output=True)

    threading.Thread(target=_run, daemon=True).start()


def draw_phrase_overlay(frame: np.ndarray, words: List[str]) -> np.ndarray:
    """Draw a semi-transparent bottom bar showing the accumulated phrase."""
    if not words:
        return frame

    h, w = frame.shape[:2]
    bar_height = 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    text = " ".join(words)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.6
    thickness = 3
    text_w, text_h = cv2.getTextSize(text, font, font_scale, thickness)[0]

    if text_w > w - 40:
        font_scale *= (w - 40) / text_w
        text_w, text_h = cv2.getTextSize(text, font, font_scale, thickness)[0]

    x = (w - text_w) // 2
    y = h - bar_height // 2 + text_h // 2
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", label.lower())


def format_label(label: str) -> str:
    if normalize_label(label) == "iloveyou":
        return "I LOVE YOU"
    if normalize_label(label) == "thankyou":
        return "THANK YOU"
    return label.upper()


def load_sign_colors(path: Path) -> Dict[str, Tuple[int, int, int]]:
    if not path.exists():
        return {}
    payload = load_json_file(str(path))

    labels = payload.get("classes", [])
    colors = payload.get("colors", [])
    color_map: Dict[str, Tuple[int, int, int]] = {}
    for label, color in zip(labels, colors):
        if not isinstance(color, list) or len(color) != 3:
            continue
        color_map[normalize_label(label)] = (int(color[0]), int(color[1]), int(color[2]))
    return color_map


def bounding_box(points_xy: np.ndarray, pad: int = 16) -> Tuple[int, int, int, int]:
    min_xy = np.floor(points_xy.min(axis=0)).astype(int)
    max_xy = np.ceil(points_xy.max(axis=0)).astype(int)
    x1, y1 = int(min_xy[0]) - pad, int(min_xy[1]) - pad
    x2, y2 = int(max_xy[0]) + pad, int(max_xy[1]) + pad
    return x1, y1, x2, y2


def finger_states(hand_points: np.ndarray, handedness: str) -> Dict[str, bool]:
    """
    Open/closed heuristics for each finger.
    For index/middle/ring/pinky: compare tip-y with pip-y.
    For thumb: compare x direction based on handedness and wrist distance.
    """
    wrist = hand_points[0]
    index_mcp = hand_points[5]
    pinky_mcp = hand_points[17]
    palm_scale = max(np.linalg.norm(index_mcp[:2] - pinky_mcp[:2]), 1e-6)

    index_open = hand_points[8][1] < hand_points[6][1]
    middle_open = hand_points[12][1] < hand_points[10][1]
    ring_open = hand_points[16][1] < hand_points[14][1]
    pinky_open = hand_points[20][1] < hand_points[18][1]

    thumb_tip = hand_points[4]
    thumb_ip = hand_points[3]
    if handedness == "right":
        thumb_open = (thumb_tip[0] - thumb_ip[0]) > 0.15 * palm_scale
    else:
        thumb_open = (thumb_ip[0] - thumb_tip[0]) > 0.15 * palm_scale
    thumb_far_from_wrist = np.linalg.norm(thumb_tip[:2] - wrist[:2]) > 0.45 * palm_scale
    thumb_open = bool(thumb_open or thumb_far_from_wrist)

    return {
        "thumb": bool(thumb_open),
        "index": bool(index_open),
        "middle": bool(middle_open),
        "ring": bool(ring_open),
        "pinky": bool(pinky_open),
    }


def classify_shape_sign(
    hand_points: np.ndarray,
    handedness: str,
    face_landmarks,
    image_shape: Tuple[int, int, int],
) -> Tuple[Optional[str], float]:
    states = finger_states(hand_points, handedness)
    open_count = sum(states.values())

    index_tip = hand_points[8]
    thumb_tip = hand_points[4]
    pinky_tip = hand_points[20]
    wrist = hand_points[0]
    palm_width = max(np.linalg.norm(hand_points[5][:2] - hand_points[17][:2]), 1e-6)
    palm_height = max(np.linalg.norm(wrist[:2] - hand_points[9][:2]), 1e-6)
    palm_size = max(palm_width, palm_height)

    if palm_size < MIN_PALM_SIZE_PX:
        return None, 0.0

    # I LOVE YOU: thumb/index/pinky open, middle/ring closed.
    if states["thumb"] and states["index"] and states["pinky"] and not states["middle"] and not states["ring"]:
        return "iloveyou", 0.92

    # YES: closed fist.
    if open_count == 0:
        return "yes", 0.78

    # STOP: all fingers open and hand is broad/open.
    if open_count >= 4 and np.linalg.norm(index_tip[:2] - pinky_tip[:2]) > 0.7 * palm_width:
        # THANK YOU: open hand near mouth/chin.
        if face_landmarks is not None:
            h, w = image_shape[:2]
            mouth = face_landmarks.landmark[13]
            mouth_xy = np.array([mouth.x * w, mouth.y * h], dtype=np.float32)
            dist_to_mouth = np.linalg.norm(index_tip[:2] - mouth_xy)
            if dist_to_mouth < 1.1 * palm_size:
                return "thankyou", 0.86
        # HELLO: default open upright hand.
        if index_tip[1] < wrist[1]:
            return "hello", 0.8
        return "stop", 0.72

    # NO: index + middle open close together, others mostly closed.
    gap_im = np.linalg.norm(hand_points[8][:2] - hand_points[12][:2])
    if states["index"] and states["middle"] and not states["ring"] and not states["pinky"] and gap_im < 0.45 * palm_width:
        return "no", 0.74

    # LOVE: both index and thumb open while others mostly closed.
    if states["thumb"] and states["index"] and not states["middle"] and not states["ring"] and not states["pinky"]:
        if np.linalg.norm(index_tip[:2] - thumb_tip[:2]) > 0.7 * palm_width:
            return "love", 0.68

    # PLEASE: open hand but not near mouth (weak proxy).
    if open_count >= 4:
        return "please", 0.62

    return None, 0.0


def draw_box_with_label(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int],
) -> None:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    text = format_label(label)
    text_scale = 0.75
    text_thickness = 2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
    tx = max(0, min(x1, w - tw - 8))
    ty = min(h - 6, y2 + th + 8)

    cv2.rectangle(image, (tx - 4, ty - th - 6), (tx + tw + 4, ty + 4), color, -1)
    cv2.putText(image, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)


def detect_and_draw(
    image: np.ndarray, results, sign_colors: Dict[str, Tuple[int, int, int]]
) -> Tuple[Optional[str], float]:
    runtime = detect_and_draw.runtime_model
    best_label: Optional[str] = None
    best_conf = 0.0

    def consider(label: Optional[str], conf: float) -> None:
        nonlocal best_label, best_conf
        if label is not None and conf >= 0.6 and conf > best_conf:
            best_label = label
            best_conf = conf

    if runtime and runtime.is_ready and runtime.uses_dual_hand:
        label, conf, left_ok, right_ok = runtime.predict_frame(
            results.left_hand_landmarks,
            results.right_hand_landmarks,
            results.face_landmarks,
            image.shape,
        )

        hand_sets = []
        if left_ok and results.left_hand_landmarks is not None:
            hand_sets.append((results.left_hand_landmarks, "left"))
        if right_ok and results.right_hand_landmarks is not None:
            hand_sets.append((results.right_hand_landmarks, "right"))

        for hand_landmarks, _ in hand_sets:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(227, 224, 113), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(227, 224, 113), thickness=2, circle_radius=2),
            )

        if label is None or conf < 0.6 or not hand_sets:
            return best_label, best_conf

        all_points = []
        for hand_landmarks, _ in hand_sets:
            all_points.append(hand_to_pixels(hand_landmarks, image.shape)[:, :2])
        points_xy = np.vstack(all_points)
        box = bounding_box(points_xy)
        color = sign_colors.get(normalize_label(label), (0, 180, 255))
        draw_box_with_label(image, box, label, color)
        consider(label, conf)
        return best_label, best_conf

    for hand_landmarks, handedness in (
        (results.left_hand_landmarks, "left"),
        (results.right_hand_landmarks, "right"),
    ):
        if hand_landmarks is None:
            continue

        hand_points = hand_to_pixels(hand_landmarks, image.shape)
        label: Optional[str] = None
        conf = 0.0
        if runtime and runtime.is_ready:
            label, conf = runtime.predict_hand(hand_points, handedness, results.face_landmarks, image.shape)
        if label is None:
            label, conf = classify_shape_sign(hand_points, handedness, results.face_landmarks, image.shape)

        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(227, 224, 113), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(227, 224, 113), thickness=2, circle_radius=2),
        )

        if label is None or conf < 0.6:
            continue

        box = bounding_box(hand_points[:, :2])
        color = sign_colors.get(normalize_label(label), (0, 180, 255))
        draw_box_with_label(image, box, label, color)
        consider(label, conf)

    return best_label, best_conf


detect_and_draw.runtime_model: Optional[RuntimeSignClassifier] = None


def real_time_asl() -> None:
    sign_colors = load_sign_colors(SIGNS_FILE)
    runtime_model = RuntimeSignClassifier(model_path=MODEL_FILE, min_confidence=0.55, margin=0.10)
    detect_and_draw.runtime_model = runtime_model
    cap = cv2.VideoCapture(0)
    print("[INFO] Starting webcam sign detection...")
    if not cap.isOpened():
        print("[ERROR] Could not open webcam (index 0). Close other camera apps and try again.")
        return

    phrase_words: List[str] = []
    last_detection_time = 0.0
    current_sign: Optional[str] = None
    current_sign_start = 0.0
    last_added_word: Optional[str] = None
    tts_spoken = False

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Webcam opened but no frame was read. Exiting.")
                break

            now = time.time()

            # Mirror first so MediaPipe landmarks and display use the same orientation.
            frame = cv2.flip(frame, 1)
            image, results = mediapipe_detection(frame, holistic)

            label, _ = detect_and_draw(image, results, sign_colors)

            if label is not None:
                display_word = format_label(label)
                last_detection_time = now
                tts_spoken = False

                if display_word == current_sign:
                    if (
                        now - current_sign_start >= SIGN_CONFIRM_DURATION
                        and display_word != last_added_word
                    ):
                        phrase_words.append(display_word)
                        last_added_word = display_word
                else:
                    current_sign = display_word
                    current_sign_start = now
            else:
                current_sign = None
                current_sign_start = 0.0

                if phrase_words and last_detection_time > 0:
                    silence = now - last_detection_time

                    if silence >= PHRASE_CLEAR_TIMEOUT - TTS_TRIGGER_OFFSET and not tts_spoken:
                        speak(" ".join(phrase_words))
                        tts_spoken = True

                    if silence >= PHRASE_CLEAR_TIMEOUT:
                        phrase_words.clear()
                        last_added_word = None
                        tts_spoken = False

            image = draw_phrase_overlay(image, phrase_words)

            cv2.putText(
                image,
                "Press Q to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            mode_text = "Model: trained classifier" if runtime_model.is_ready else "Model: heuristic fallback"
            cv2.putText(
                image,
                mode_text,
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (210, 210, 210),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Webcam Feed", image)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    real_time_asl()