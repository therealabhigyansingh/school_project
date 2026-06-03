"""SignSpeak training-data ingestion: convert images and videos into landmark samples.

Usage:
    python ingest.py [--input ./inputs] [--stride 10] [--max-per-clip 100]

Expected input layout (case-insensitive letter folder names):
    <input>/
      A/
        anything.jpg, anything.png, clip.mp4, ...
      B/
        ...
      ...

For each image: detect hand landmarks, normalize, tile into a 30-frame sample.
For each video: detect landmarks frame-by-frame, then slide a 30-frame window across
each contiguous-hand run with the given stride.

Outputs land in trainer/data/{LETTER}/ingest_{source-stem}_{idx}.npy and mix freely
with samples produced by capture.py. After ingest, run `python train.py`.

Tip: a public ASL alphabet dataset (e.g. Kaggle "ASL Alphabet" by grassknoted) can be
dropped straight into inputs/ and trained from end-to-end.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

# Make Windows consoles tolerate unicode in our progress prints (cp1252 default chokes on →/✓).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from inference.landmarks import LandmarkExtractor  # noqa: E402

DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "inputs"
DATA_DIR = Path(__file__).resolve().parent / "data"
SEQUENCE_LEN = 30
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".webm", ".mkv", ".m4v"}
LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def ingest_image(extractor, path: Path) -> list[np.ndarray]:
    img = cv2.imread(str(path))
    if img is None:
        return []
    _raw, normalized = extractor.extract(img)
    if normalized is None:
        return []
    return [np.tile(normalized[None, ...], (SEQUENCE_LEN, 1, 1))]


def ingest_video(extractor, path: Path, stride: int, max_per_clip: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []

    runs: list[list[np.ndarray]] = [[]]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        _raw, normalized = extractor.extract(frame)
        if normalized is None:
            if runs[-1]:
                runs.append([])
        else:
            runs[-1].append(normalized)
    cap.release()

    samples = []
    for run in runs:
        if len(run) < SEQUENCE_LEN:
            continue
        end = len(run) - SEQUENCE_LEN + 1
        for start in range(0, end, max(1, stride)):
            samples.append(np.stack(run[start:start + SEQUENCE_LEN]))
            if len(samples) >= max_per_clip:
                return samples
    return samples


def save_samples(letter: str, source_stem: str, samples: list[np.ndarray]):
    out_dir = DATA_DIR / letter
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = "".join(c if c.isalnum() else "_" for c in source_stem)[:60]
    for i, seq in enumerate(samples):
        path = out_dir / f"ingest_{safe_stem}_{i:04d}.npy"
        np.save(path, seq)


def iter_letter_dirs(input_dir: Path) -> Iterator[Path]:
    """Walk recursively and yield every directory whose name is a single A-Z letter.

    This handles both flat layouts (inputs/A/...) and nested ones like Kaggle's
    asl_alphabet_train/asl_alphabet_train/A/... — drop the unzipped dataset anywhere
    inside inputs/ and it'll be found.
    """
    seen = set()
    for path in sorted(input_dir.rglob("*")):
        if not path.is_dir():
            continue
        if path.name.upper() not in LETTERS:
            continue
        # Dedupe by absolute path
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        yield path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR,
                        help=f"input directory (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--stride", type=int, default=10,
                        help="frames between video samples (lower = more samples, more correlated)")
    parser.add_argument("--max-per-clip", type=int, default=100,
                        help="max samples to extract from one video")
    args = parser.parse_args()

    input_dir: Path = args.input
    if not input_dir.exists():
        print(f"ERROR: {input_dir} does not exist.")
        print("Create it and add per-letter subfolders, e.g. inputs/A/photo.jpg")
        sys.exit(1)

    letter_dirs = list(iter_letter_dirs(input_dir))
    if not letter_dirs:
        print(f"No letter subfolders found in {input_dir}.")
        print("Expected layout: inputs/A/*.jpg, inputs/B/*.mp4, ...")
        sys.exit(1)

    extractor = LandmarkExtractor()
    counts: dict[str, int] = defaultdict(int)
    skipped: dict[str, int] = defaultdict(int)

    try:
        for i, letter_dir in enumerate(letter_dirs):
            letter = letter_dir.name.upper()
            files = [
                f for f in sorted(letter_dir.iterdir())
                if f.is_file() and f.suffix.lower() in (IMAGE_EXTS | VIDEO_EXTS)
            ]
            if not files:
                continue

            n_img = sum(1 for f in files if f.suffix.lower() in IMAGE_EXTS)
            n_vid = sum(1 for f in files if f.suffix.lower() in VIDEO_EXTS)
            print(f"[{i+1}/{len(letter_dirs)}] {letter}: {n_img} images, "
                  f"{n_vid} videos", end="", flush=True)

            produced = 0
            for f in files:
                if f.suffix.lower() in IMAGE_EXTS:
                    samples = ingest_image(extractor, f)
                else:
                    samples = ingest_video(extractor, f, args.stride, args.max_per_clip)

                if samples:
                    save_samples(letter, f.stem, samples)
                    produced += len(samples)
                else:
                    skipped[letter] += 1

            counts[letter] = produced
            print(f"  →  {produced} samples")
    finally:
        extractor.close()

    total = sum(counts.values())
    total_skipped = sum(skipped.values())
    print(f"\nDone. {total} samples produced across {len(counts)} letters.")
    if total_skipped:
        print(f"Skipped {total_skipped} files (no hand detected).")
    print("\nNext: python train.py")


if __name__ == "__main__":
    main()
