"""
Feature extraction utilities for static hand-shape sign classification.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def normalize_label(label: str) -> str:
    return "".join(ch for ch in label.lower() if ch.isalnum())


def hand_to_pixels(hand_landmarks, image_shape: tuple[int, int, int]) -> np.ndarray:
    h, w = image_shape[:2]
    return np.array([[lm.x * w, lm.y * h, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)


def _safe_unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.zeros_like(v)
    return v / n


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in radians from three 2D points."""
    ba = _safe_unit(a - b)
    bc = _safe_unit(c - b)
    dot = float(np.clip(np.dot(ba, bc), -1.0, 1.0))
    return float(np.arccos(dot))


def finger_states(hand_points: np.ndarray, handedness: str) -> dict[str, bool]:
    wrist = hand_points[0]
    index_mcp = hand_points[5]
    pinky_mcp = hand_points[17]
    palm_scale = max(float(np.linalg.norm(index_mcp[:2] - pinky_mcp[:2])), 1e-6)

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


def build_hand_features(
    hand_points: np.ndarray,
    handedness: str,
    face_landmarks: Optional[object],
    image_shape: tuple[int, int, int],
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Build a compact, scale-invariant feature vector from one hand.
    """
    wrist_xy = hand_points[0][:2]
    index_mcp_xy = hand_points[5][:2]
    pinky_mcp_xy = hand_points[17][:2]
    palm_width = max(float(np.linalg.norm(index_mcp_xy - pinky_mcp_xy)), 1e-6)

    # Relative hand coordinates (21 x 2) normalized by palm width.
    rel = (hand_points[:, :2] - wrist_xy) / palm_width

    # Finger state booleans.
    states = finger_states(hand_points, handedness)
    open_count = float(sum(states.values()))

    # Finger bend angles: MCP/PIP chain for index/middle/ring/pinky.
    idx_angle = _angle(hand_points[5][:2], hand_points[6][:2], hand_points[8][:2])
    mid_angle = _angle(hand_points[9][:2], hand_points[10][:2], hand_points[12][:2])
    ring_angle = _angle(hand_points[13][:2], hand_points[14][:2], hand_points[16][:2])
    pnk_angle = _angle(hand_points[17][:2], hand_points[18][:2], hand_points[20][:2])

    # Spread and orientation proxies.
    index_tip_xy = hand_points[8][:2]
    pinky_tip_xy = hand_points[20][:2]
    thumb_tip_xy = hand_points[4][:2]
    wrist_to_middle_tip = hand_points[12][:2] - wrist_xy
    palm_upright = float(-_safe_unit(wrist_to_middle_tip)[1])  # higher -> more upright
    spread = float(np.linalg.norm(index_tip_xy - pinky_tip_xy) / palm_width)
    thumb_index_dist = float(np.linalg.norm(thumb_tip_xy - index_tip_xy) / palm_width)

    # Hand-face distance (mouth region) if available.
    mouth_dist = 3.0
    if face_landmarks is not None:
        h, w = image_shape[:2]
        mouth = face_landmarks.landmark[13]
        mouth_xy = np.array([mouth.x * w, mouth.y * h], dtype=np.float32)
        mouth_dist = float(np.linalg.norm(index_tip_xy - mouth_xy) / palm_width)

    handedness_num = 1.0 if handedness == "right" else 0.0
    bool_feats = np.array(
        [
            float(states["thumb"]),
            float(states["index"]),
            float(states["middle"]),
            float(states["ring"]),
            float(states["pinky"]),
            open_count,
            handedness_num,
        ],
        dtype=np.float32,
    )
    geom_feats = np.array(
        [
            idx_angle,
            mid_angle,
            ring_angle,
            pnk_angle,
            spread,
            thumb_index_dist,
            palm_upright,
            mouth_dist,
        ],
        dtype=np.float32,
    )

    # Final vector: normalized coords + booleans + geometric features.
    vector = np.concatenate([rel.flatten(), bool_feats, geom_feats]).astype(np.float32)
    metrics = {
        "palm_width": palm_width,
        "open_count": open_count,
        "spread": spread,
        "mouth_dist": mouth_dist,
        "palm_upright": palm_upright,
    }
    return vector, metrics


FEATURES_PER_HAND = 21 * 2 + 7 + 8  # rel coords + finger/geom features


def extract_hand_features(
    hand_landmarks,
    handedness: str,
    face_landmarks,
    image_shape: tuple[int, int, int],
    min_palm_width: float = 40.0,
) -> tuple[np.ndarray, bool, np.ndarray | None]:
    """Return (feature_vector, is_valid, hand_points_pixels)."""
    if hand_landmarks is None:
        return np.zeros((FEATURES_PER_HAND,), dtype=np.float32), False, None

    hand_points = hand_to_pixels(hand_landmarks, image_shape)
    feats, metrics = build_hand_features(hand_points, handedness, face_landmarks, image_shape)
    if metrics["palm_width"] < min_palm_width:
        return np.zeros((FEATURES_PER_HAND,), dtype=np.float32), False, hand_points
    return feats.astype(np.float32), True, hand_points


def build_frame_features(
    left_hand_landmarks,
    right_hand_landmarks,
    face_landmarks,
    image_shape: tuple[int, int, int],
    min_palm_width: float = 40.0,
) -> tuple[np.ndarray, bool, bool, np.ndarray | None, np.ndarray | None]:
    """
    Build a dual-hand feature vector: [left_feats, right_feats].
    Missing hands are zero-filled; left_present/right_present are returned separately.
    """
    left_feats, left_ok, left_points = extract_hand_features(
        left_hand_landmarks, "left", face_landmarks, image_shape, min_palm_width
    )
    right_feats, right_ok, right_points = extract_hand_features(
        right_hand_landmarks, "right", face_landmarks, image_shape, min_palm_width
    )
    vector = np.concatenate([left_feats, right_feats]).astype(np.float32)
    return vector, left_ok, right_ok, left_points, right_points

