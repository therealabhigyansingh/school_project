# inputs/

Drop images and videos here, organized by letter, then run `python ../ingest.py` to
convert them into `.npy` training samples in `../data/`.

## Layout

```
inputs/
  A/
    photo1.jpg
    photo2.png
    clip1.mp4
  B/
    ...
  ...
```

- Letter folder names are case-insensitive (`a/` and `A/` both work).
- Supported image formats: `.jpg .jpeg .png .bmp .webp`
- Supported video formats: `.mp4 .avi .mov .webm .mkv .m4v`
- Files where MediaPipe can't detect a hand are skipped (logged at the end).

## Public datasets that work out-of-the-box

Drop them into per-letter subfolders and ingest:

- **Kaggle: ASL Alphabet (grassknoted)** — 87k still images, 26 letters + 3 extras. Just delete the `del`/`nothing`/`space` folders before ingest.
- **Kaggle: Sign Language MNIST** — small images, 24 letters (no J, Z).
- Any video collection where each clip is one letter.

## Mixing with live captures

Ingested samples and `capture.py` samples live in the same `data/{LETTER}/` folders and
are treated identically by `train.py`. You can mix them however you like.
