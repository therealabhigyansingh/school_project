"""SignSpeak training-data capture tool.

Usage:
    python capture.py [--target 150] [--stride 10]

Keys:
    A-Z       set target letter
    SPACE     record one 30-frame sequence (manual, one-shot)
    B         toggle BURST mode — sliding-window stream that saves a sample every
              `--stride` frames while a hand is detected
    S         print sample counts per letter
    Q / ESC   quit

In BURST mode the script keeps a rolling 30-frame buffer of your hand and saves a
snapshot every `stride` frames. Move your hand continuously while burst is active —
shift position, distance, angle. Variation across snapshots is what gets you to 90%.

Saves per-sample arrays to trainer/data/{LETTER}/{timestamp}.npy
Each sample is a (30, 21, 3) array of normalized landmarks.
"""
import argparse
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

# Make Windows consoles tolerate unicode in our progress prints (cp1252 default chokes on ✓/→/█).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv2
import numpy as np

# Reuse the backend's landmark module so training and inference normalize identically.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from inference.landmarks import LandmarkExtractor  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
SEQUENCE_LEN = 30

ACCENT = (106, 255, 60)  # BGR
DIM = (200, 200, 200)
HOT = (60, 100, 255)
DARK = (40, 40, 40)
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def count_samples():
    counts = defaultdict(int)
    if DATA_DIR.exists():
        for letter_dir in DATA_DIR.iterdir():
            if letter_dir.is_dir():
                counts[letter_dir.name.upper()] = len(list(letter_dir.glob("*.npy")))
    return counts


def save_sample(letter: str, sequence: np.ndarray):
    out_dir = DATA_DIR / letter
    out_dir.mkdir(parents=True, exist_ok=True)
    # Use ns precision so we never collide on rapid stream saves
    path = out_dir / f"{time.time_ns()}.npy"
    np.save(path, sequence)


def draw_progress_bar(frame, x, y, w, h, progress, color, bg=DARK):
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1)
    fill = int(w * max(0.0, min(1.0, progress)))
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + h), color, -1)


def draw_hud(frame, *, target_letter, target_count, counts, mode, manual_buf_len,
             rolling_buf_len, stride, frames_since_save, flash_until):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    ltr = target_letter or "—"
    count = counts.get(target_letter, 0) if target_letter else 0
    cv2.putText(frame, f"target: {ltr}", (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, ACCENT, 2)
    cv2.putText(frame, f"{count} / {target_count}", (16, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, DIM, 2)

    if target_letter:
        draw_progress_bar(frame, 160, 16, w - 176, 14,
                          count / max(1, target_count), ACCENT)

    # Mode badge
    if mode == "manual":
        cv2.putText(frame, f"REC  {manual_buf_len}/{SEQUENCE_LEN}", (160, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, HOT, 2)
    elif mode == "burst":
        if rolling_buf_len < SEQUENCE_LEN:
            cv2.putText(frame, f"BURST  warm-up {rolling_buf_len}/{SEQUENCE_LEN}",
                        (160, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, HOT, 2)
        else:
            ticks_to_save = stride - frames_since_save
            cv2.putText(frame,
                        f"BURST  stride {stride}  next in {ticks_to_save}f  — keep moving!",
                        (160, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ACCENT, 2)

    done_letters = sum(1 for c in counts.values() if c >= target_count)
    total = sum(counts.values())
    grand_target = target_count * len(LETTERS)
    cv2.putText(frame, f"total: {total} / {grand_target}   "
                       f"complete: {done_letters} / 26",
                (16, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIM, 1)
    draw_progress_bar(frame, 16, 102, w - 32, 8,
                      total / max(1, grand_target), ACCENT)

    legend = "A-Z: pick letter   SPACE: 1 sample   B: burst   S: counts   Q: quit"
    cv2.putText(frame, legend, (16, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, DIM, 1)

    if time.time() < flash_until:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), ACCENT, 6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=150,
                        help="target sample count per letter (default 150)")
    parser.add_argument("--stride", type=int, default=10,
                        help="frames between saves in burst mode "
                             "(default 10; lower = faster but more correlated samples)")
    args = parser.parse_args()
    target_count = args.target
    stride = max(1, args.stride)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open webcam (index 0).")
        sys.exit(1)

    extractor = LandmarkExtractor()
    target_letter = None
    mode = "idle"  # 'idle' | 'manual' | 'burst'
    manual_buffer: list[np.ndarray] = []
    rolling: deque = deque(maxlen=SEQUENCE_LEN)
    frames_since_save = 0
    counts = count_samples()
    flash_until = 0.0

    print(f"Capture ready. Target {target_count} samples per letter "
          f"({target_count * 26} total). Burst stride: {stride} frames.")
    print("Press a letter A-Z to set target, then SPACE for one sample or B for burst.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            raw, normalized = extractor.extract(frame)

            now = time.time()

            if raw is not None:
                fh, fw = frame.shape[:2]
                for x, y, _z in raw:
                    cv2.circle(frame, (int(x * fw), int(y * fh)), 4, ACCENT, -1)

            # ---- recording logic ----
            if normalized is None:
                # hand lost — reset rolling buffer so the next save isn't a partial transition
                rolling.clear()
                frames_since_save = 0
                if mode == "manual":
                    manual_buffer.clear()  # restart manual on hand loss
            else:
                if mode == "manual":
                    manual_buffer.append(normalized)
                    if len(manual_buffer) >= SEQUENCE_LEN:
                        seq = np.stack(manual_buffer)
                        save_sample(target_letter, seq)
                        counts[target_letter] = counts.get(target_letter, 0) + 1
                        manual_buffer = []
                        flash_until = now + 0.25
                        mode = "idle"

                elif mode == "burst":
                    rolling.append(normalized)
                    if len(rolling) == SEQUENCE_LEN:
                        frames_since_save += 1
                        if frames_since_save >= stride:
                            seq = np.stack(rolling)
                            save_sample(target_letter, seq)
                            counts[target_letter] = counts.get(target_letter, 0) + 1
                            frames_since_save = 0
                            flash_until = now + 0.08  # subtle flash, don't strobe
                            if counts[target_letter] >= target_count:
                                mode = "idle"
                                rolling.clear()
                                flash_until = now + 0.8
                                print(f"\n  ✓ {target_letter}: {counts[target_letter]} "
                                      f"samples (target reached). Pick the next letter.")

            draw_hud(frame,
                     target_letter=target_letter,
                     target_count=target_count,
                     counts=counts,
                     mode=mode,
                     manual_buf_len=len(manual_buffer),
                     rolling_buf_len=len(rolling),
                     stride=stride,
                     frames_since_save=frames_since_save,
                     flash_until=flash_until)

            cv2.imshow("SignSpeak Capture", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                if target_letter is not None and mode != "manual":
                    manual_buffer = []
                    rolling.clear()
                    mode = "manual"
            elif key == ord('b'):
                if target_letter is None:
                    print("Pick a target letter first (A-Z).")
                    continue
                if mode == "burst":
                    mode = "idle"
                    rolling.clear()
                    frames_since_save = 0
                else:
                    mode = "burst"
                    rolling.clear()
                    frames_since_save = 0
                    manual_buffer = []
            elif key == ord('s'):
                print("\nSample counts:")
                for ltr in LETTERS:
                    n = counts.get(ltr, 0)
                    bar = "█" * int(20 * min(1.0, n / target_count))
                    bar = bar.ljust(20, "·")
                    marker = "✓" if n >= target_count else " "
                    print(f"  {marker} {ltr}: {n:4d} / {target_count}  {bar}")
                total = sum(counts.values())
                print(f"  total: {total} / {target_count * 26}\n")
            elif 65 <= key <= 90 or 97 <= key <= 122:
                target_letter = chr(key).upper()
                mode = "idle"
                manual_buffer = []
                rolling.clear()
                frames_since_save = 0
    finally:
        extractor.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
