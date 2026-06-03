import time

from ._base import BaseSession

NO_HAND_FOR_SPACE_S = 5.0


class DecoderSession(BaseSession):
    def __init__(self, predictor):
        super().__init__(predictor)
        self.current_word = ""
        self.full_text = ""
        self.last_hand_ts = time.time()

    def process_frame(self, jpeg_bytes):
        raw, prediction, confidence, committed = self._process(jpeg_bytes)
        now = time.time()
        hand_present = raw is not None

        if hand_present:
            self.last_hand_ts = now

        if committed is not None:
            self.current_word += committed

        no_hand_seconds = 0.0 if hand_present else (now - self.last_hand_ts)
        if no_hand_seconds >= NO_HAND_FOR_SPACE_S and self.current_word:
            self.full_text += self.current_word + " "
            self.current_word = ""
            self.last_hand_ts = now  # reset so we don't fire repeatedly

        return {
            "landmarks": self._serialize_landmarks(raw),
            "prediction": prediction,
            "confidence": confidence,
            "mode": {
                "current_word": self.current_word,
                "full_text": self.full_text,
                "hand_present": hand_present,
                "no_hand_seconds": round(no_hand_seconds, 1),
            },
        }

    def handle_control(self, msg):
        if msg.get("action") == "reset":
            self.current_word = ""
            self.full_text = ""
            self.last_hand_ts = time.time()
            return {
                "landmarks": None,
                "prediction": None,
                "confidence": 0.0,
                "mode": {
                    "current_word": "",
                    "full_text": "",
                    "hand_present": False,
                    "no_hand_seconds": 0.0,
                },
            }
        return None
