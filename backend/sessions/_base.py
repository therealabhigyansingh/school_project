"""Shared per-WebSocket session logic: landmark extraction, sequence buffer, debounced commit."""
from collections import deque

import numpy as np

from inference.landmarks import LandmarkExtractor, decode_jpeg

BUFFER_SIZE = 30
CONFIDENCE_THRESHOLD = 0.85
HOLD_FRAMES_TO_COMMIT = 5


class BaseSession:
    def __init__(self, predictor):
        self.predictor = predictor
        self.extractor = LandmarkExtractor()
        self.buffer = deque(maxlen=BUFFER_SIZE)
        self._consecutive_letter = None
        self._consecutive_count = 0
        self._last_committed = None

    def _process(self, jpeg_bytes):
        """Run extraction + prediction. Returns (raw, prediction, confidence, committed_letter_or_None).

        - `raw` is the (21, 3) image-space landmark array (None when no hand)
        - `committed_letter` fires once when the user holds the same letter for HOLD_FRAMES_TO_COMMIT
          consecutive frames at confidence >= CONFIDENCE_THRESHOLD; resets on hand release or letter change.
        """
        frame = decode_jpeg(jpeg_bytes)
        raw, normalized = self.extractor.extract(frame)

        if normalized is None:
            self.buffer.clear()
            self._consecutive_count = 0
            self._consecutive_letter = None
            self._last_committed = None
            return None, None, 0.0, None

        self.buffer.append(normalized)
        prediction, confidence = None, 0.0
        committed = None

        if len(self.buffer) == BUFFER_SIZE:
            seq = np.stack(self.buffer).reshape(BUFFER_SIZE, -1)
            prediction, confidence = self.predictor.predict(seq)

            if prediction is not None and confidence >= CONFIDENCE_THRESHOLD:
                if prediction == self._consecutive_letter:
                    self._consecutive_count += 1
                else:
                    self._consecutive_letter = prediction
                    self._consecutive_count = 1

                if (
                    self._consecutive_count >= HOLD_FRAMES_TO_COMMIT
                    and prediction != self._last_committed
                ):
                    committed = prediction
                    self._last_committed = prediction
            else:
                self._consecutive_count = 0

        return raw, prediction, confidence, committed

    @staticmethod
    def _serialize_landmarks(raw):
        return None if raw is None else raw.tolist()

    def handle_control(self, msg):
        return None

    def close(self):
        self.extractor.close()
