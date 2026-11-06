"""
Runtime wrapper for a trained static hand-shape classifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from src.sign_features import FEATURES_PER_HAND, build_frame_features, build_hand_features


class RuntimeSignClassifier:
    def __init__(
        self,
        model_path: Path = Path("models/handshape_classifier.joblib"),
        min_confidence: float = 0.55,
        margin: float = 0.10,
    ) -> None:
        self.model_path = model_path
        self.min_confidence = float(min_confidence)
        self.margin = float(margin)
        self.model = None
        self.classes: list[str] = []
        self.feature_layout = "single_hand"
        self.features_per_hand = FEATURES_PER_HAND
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            return
        payload = joblib.load(self.model_path)
        self.model = payload["model"]
        self.classes = list(payload["classes"])
        self.feature_layout = payload.get("feature_layout", "single_hand")
        self.features_per_hand = int(payload.get("features_per_hand", FEATURES_PER_HAND))

    @property
    def is_ready(self) -> bool:
        return self.model is not None and len(self.classes) > 0

    @property
    def uses_dual_hand(self) -> bool:
        return self.feature_layout == "dual_hand"

    def _apply_threshold(self, probs: np.ndarray) -> tuple[Optional[str], float]:
        best_idx = int(np.argmax(probs))
        best = float(probs[best_idx])

        if len(probs) > 1:
            sorted_probs = np.sort(probs)
            second_best = float(sorted_probs[-2])
        else:
            second_best = 0.0
        margin = best - second_best

        if best < self.min_confidence or margin < self.margin:
            return None, best
        return self.classes[best_idx], best

    def predict_hand(
        self,
        hand_points: np.ndarray,
        handedness: str,
        face_landmarks,
        image_shape: tuple[int, int, int],
    ) -> tuple[Optional[str], float]:
        if not self.is_ready or self.uses_dual_hand:
            return None, 0.0

        feats, _ = build_hand_features(hand_points, handedness, face_landmarks, image_shape)
        probs = self.model.predict_proba(feats.reshape(1, -1))[0]
        return self._apply_threshold(probs)

    def predict_frame(
        self,
        left_hand_landmarks,
        right_hand_landmarks,
        face_landmarks,
        image_shape: tuple[int, int, int],
    ) -> tuple[Optional[str], float, bool, bool]:
        if not self.is_ready:
            return None, 0.0, False, False

        vector, left_ok, right_ok, _, _ = build_frame_features(
            left_hand_landmarks,
            right_hand_landmarks,
            face_landmarks,
            image_shape,
        )

        if self.uses_dual_hand:
            probs = self.model.predict_proba(vector.reshape(1, -1))[0]
            label, conf = self._apply_threshold(probs)
            return label, conf, left_ok, right_ok

        # Legacy single-hand model: pick the best hand prediction.
        best_label: Optional[str] = None
        best_conf = 0.0
        for hand_landmarks, handedness in (
            (left_hand_landmarks, "left"),
            (right_hand_landmarks, "right"),
        ):
            if hand_landmarks is None:
                continue
            hand_points = np.array(
                [[lm.x * image_shape[1], lm.y * image_shape[0], lm.z] for lm in hand_landmarks.landmark],
                dtype=np.float32,
            )
            label, conf = self.predict_hand(hand_points, handedness, face_landmarks, image_shape)
            if conf > best_conf:
                best_label, best_conf = label, conf
        return best_label, best_conf, left_ok, right_ok
