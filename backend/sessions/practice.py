import time

from ._base import BaseSession, CONFIDENCE_THRESHOLD

SUCCESS_HOLD_S = 5.0


class PracticeSession(BaseSession):
    def __init__(self, predictor):
        super().__init__(predictor)
        self.target_letter = None
        self.hold_start_ts = None
        self.success = False

    def process_frame(self, jpeg_bytes):
        raw, prediction, confidence, _committed = self._process(jpeg_bytes)
        now = time.time()
        hold_progress = 0.0

        if self.target_letter is not None and not self.success:
            matching = (
                prediction == self.target_letter
                and confidence >= CONFIDENCE_THRESHOLD
            )
            if matching:
                if self.hold_start_ts is None:
                    self.hold_start_ts = now
                hold_progress = min((now - self.hold_start_ts) / SUCCESS_HOLD_S, 1.0)
                if hold_progress >= 1.0:
                    self.success = True
            else:
                self.hold_start_ts = None

        return {
            "landmarks": self._serialize_landmarks(raw),
            "prediction": prediction,
            "confidence": confidence,
            "mode": {
                "target_letter": self.target_letter,
                "hold_progress": round(hold_progress, 3),
                "success": self.success,
            },
        }

    def handle_control(self, msg):
        action = msg.get("action")
        if action == "start":
            letter = str(msg.get("letter", "")).upper()
            if len(letter) == 1 and letter.isalpha():
                self.target_letter = letter
                self.hold_start_ts = None
                self.success = False
                self.buffer.clear()
                self._consecutive_count = 0
                self._consecutive_letter = None
                self._last_committed = None
        elif action in ("reset", "next"):
            self.target_letter = None
            self.hold_start_ts = None
            self.success = False
        return None
