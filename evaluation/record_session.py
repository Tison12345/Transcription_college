"""
Record labelled evaluation samples with the app's own microphone capture
(Silero VAD, one utterance at a time).

    # your own voice
    python -m evaluation.record_session --label genuine --count 12

    # someone else (give each person their own short name)
    python -m evaluation.record_session --label impostor --name spk01 --count 12

Files are saved as:

    eval_data/recordings/genuine/me/genuine_me_001.wav
    eval_data/recordings/impostor/spk01/impostor_spk01_001.wav

The label is the folder. Existing files are never overwritten, and
reference.wav is never touched. No speaker verification runs here.
"""

import argparse
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = PROJECT_ROOT / "eval_data" / "recordings"

# Short answers and sentences, like the follow-up call produces.
PROMPTS = [
    "yes",
    "no",
    "I am feeling okay today",
    "my pain is about three out of ten",
    "seven",
    "it hurts a little when I walk",
    "I slept badly last night",
    "yes, I would like to try that now",
    "no, not right now",
    "I took my medicine this morning",
    "can you repeat that please",
    "the swelling is better than yesterday",
]


def next_path(folder, label, name):
    """First unused <label>_<name>_NNN.wav in the folder (and below it)."""
    existing = {p.name for p in Path(folder).rglob("*.wav")}
    number = 1
    while f"{label}_{name}_{number:03d}.wav" in existing:
        number += 1
    return Path(folder) / f"{label}_{name}_{number:03d}.wav"


def record_samples(label, name, count, capture_fn=None, info=None,
                   recordings_dir=RECORDINGS_DIR, log=print):
    """
    Record `count` utterances. capture_fn / info can be injected for tests;
    by default the app's capture_one_utterance() is used (no verifier).
    """

    if capture_fn is None:
        from audio import utterance_detector as live
        capture_fn = live.capture_one_utterance
        info = live.LAST_CAPTURE_INFO

    folder = Path(recordings_dir) / label / name
    folder.mkdir(parents=True, exist_ok=True)

    saved = 0
    failures = 0

    while saved < count:
        log(f"\n[{saved + 1}/{count}] Say: {PROMPTS[saved % len(PROMPTS)]}")

        if capture_fn() is None:
            failures += 1
            log("No speech detected - try again.")
            if failures > count * 3:
                log("Too many failed attempts, stopping.")
                break
            continue

        destination = next_path(folder, label, name)
        # Copy now: the next capture overwrites the app's "latest" file.
        shutil.copyfile(info["full_path"], destination)
        saved += 1
        log(f"Saved {destination.name}")

    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--label", choices=["genuine", "impostor"],
                        required=True)
    parser.add_argument("--name", default=None,
                        help="speaker name for the folder/file (default 'me' "
                             "for genuine; required for impostor)")
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args(argv)

    name = args.name or ("me" if args.label == "genuine" else None)
    if name is None:
        parser.error("--name is required for impostor recordings")

    saved = record_samples(args.label, name, args.count)
    print(f"\nSaved {saved} recording(s).")
    print("Next: python -m evaluation.evaluate_speaker")
    return 0


if __name__ == "__main__":
    sys.exit(main())
