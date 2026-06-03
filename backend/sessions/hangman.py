import random
from pathlib import Path

from ._base import BaseSession

WORDS_FILE = Path(__file__).resolve().parent.parent / "assets" / "words.txt"
MAX_WRONG = 6


def _load_words():
    if WORDS_FILE.exists():
        words = [w.strip().upper() for w in WORDS_FILE.read_text().splitlines() if w.strip()]
        if words:
            return words
    return ["HELLO", "WORLD", "SIGN", "HAND"]


class HangmanSession(BaseSession):
    def __init__(self, predictor):
        super().__init__(predictor)
        self._words = _load_words()
        self._new_game()

    def _new_game(self):
        self.word = random.choice(self._words)
        self.revealed = ["_" for _ in self.word]
        self.wrong_guesses = []
        self.guessed = set()
        self.last_guess = None
        self.status = "playing"
        self.buffer.clear()
        self._consecutive_count = 0
        self._consecutive_letter = None
        self._last_committed = None

    def process_frame(self, jpeg_bytes):
        raw, prediction, confidence, committed = self._process(jpeg_bytes)

        if committed and self.status == "playing" and committed not in self.guessed:
            self.guessed.add(committed)
            self.last_guess = committed
            if committed in self.word:
                for i, ch in enumerate(self.word):
                    if ch == committed:
                        self.revealed[i] = ch
                if "_" not in self.revealed:
                    self.status = "won"
            else:
                self.wrong_guesses.append(committed)
                if len(self.wrong_guesses) >= MAX_WRONG:
                    self.status = "lost"

        return {
            "landmarks": self._serialize_landmarks(raw),
            "prediction": prediction,
            "confidence": confidence,
            "mode": {
                "word_mask": " ".join(self.revealed),
                "wrong_guesses": self.wrong_guesses,
                "lives_left": MAX_WRONG - len(self.wrong_guesses),
                "max_lives": MAX_WRONG,
                "last_guess": self.last_guess,
                "status": self.status,
                "word": self.word if self.status != "playing" else None,
            },
        }

    def handle_control(self, msg):
        if msg.get("action") == "new_game":
            self._new_game()
        return None
