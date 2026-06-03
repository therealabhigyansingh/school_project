"""MediaPipe wrapper + normalization. Single source of truth shared by trainer + backend."""
import cv2
import mediapipe as mp
import numpy as np

_mp_hands = mp.solutions.hands


class LandmarkExtractor:
    def __init__(self):
        self._hands = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, bgr_frame):
        """Returns (raw_landmarks (21,3) in image-relative 0..1 coords, normalized (21,3)) or (None, None)."""
        if bgr_frame is None:
            return None, None
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None, None
        hand = results.multi_hand_landmarks[0]
        raw = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand.landmark],
            dtype=np.float32,
        )
        return raw, normalize(raw)

    def close(self):
        self._hands.close()


def normalize(landmarks: np.ndarray):
    """Translate by wrist, scale by wrist→middle-finger-MCP distance. Returns (21,3) or None."""
    wrist = landmarks[0]
    centered = landmarks - wrist
    scale = float(np.linalg.norm(centered[9]))
    if scale < 1e-6:
        return None
    return (centered / scale).astype(np.float32)


def decode_jpeg(jpeg_bytes: bytes):
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
