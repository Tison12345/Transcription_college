"""
Simple speaker-verification check against reference.wav.

For every recording it prints the ECAPA cosine similarity to reference.wav,
the TARGET / NON_TARGET decision at the live threshold, and whether that
decision was correct. Then it prints a short summary.

Labels come from the folder a recording is in:

    eval_data/recordings/genuine/...    your own voice
    eval_data/recordings/impostor/...   other people

Run from the project root:

    python -m evaluation.evaluate_speaker

It only READS recordings and reference.wav. It does not change the
threshold, the live app, or any audio.
"""

import contextlib
import csv
import hashlib
import io
import sys
from pathlib import Path

from audio.rolling_verifier import SIMILARITY_THRESHOLD


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE = PROJECT_ROOT / "reference.wav"
RECORDINGS_DIR = PROJECT_ROOT / "eval_data" / "recordings"
RESULTS_CSV = PROJECT_ROOT / "eval_results" / "results.csv"

LABELS = ("genuine", "impostor")

MIN_PER_CLASS = 10      # below this, the summary prints a caution


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_recordings(recordings_dir=RECORDINGS_DIR, reference=REFERENCE):
    """
    Return [(label, path), ...] for every .wav under genuine/ and impostor/.

    Raises ValueError if a recording is a byte-for-byte copy of the
    reference (that would be a meaningless, perfect "genuine" score).
    """

    recordings_dir = Path(recordings_dir)
    reference_hash = sha256_of(reference)

    found = []
    for label in LABELS:
        for path in sorted((recordings_dir / label).rglob("*.wav")):
            if sha256_of(path) == reference_hash:
                raise ValueError(
                    f"{path} is identical to reference.wav. Remove it: the "
                    f"reference must not be used as a test recording."
                )
            found.append((label, path))

    return found


def classify(similarity, threshold=SIMILARITY_THRESHOLD):
    """Same rule as the live verifier: accepted when similarity >= threshold."""
    return "TARGET" if similarity >= threshold else "NON_TARGET"


def is_correct(label, predicted):
    """A genuine recording should be TARGET; an impostor NON_TARGET."""
    return (label == "genuine") == (predicted == "TARGET")


def evaluate(recordings, similarity_fn, threshold=SIMILARITY_THRESHOLD):
    """
    Score each (label, path). similarity_fn(path) -> cosine similarity.
    A file that cannot be scored gets an error row and is left out of the
    counts.
    """

    rows = []
    for label, path in recordings:
        row = {"file": path.name, "label": label, "similarity": None,
               "predicted": "", "correct": None, "error": ""}
        try:
            row["similarity"] = float(similarity_fn(path))
            row["predicted"] = classify(row["similarity"], threshold)
            row["correct"] = is_correct(label, row["predicted"])
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    return rows


def summarize(rows):
    scored = [r for r in rows if r["error"] == ""]
    genuine = [r for r in scored if r["label"] == "genuine"]
    impostor = [r for r in scored if r["label"] == "impostor"]

    genuine_accepted = sum(r["predicted"] == "TARGET" for r in genuine)
    impostor_accepted = sum(r["predicted"] == "TARGET" for r in impostor)
    correct = sum(r["correct"] for r in scored)

    return {
        "genuine_total": len(genuine),
        "genuine_accepted": genuine_accepted,
        "genuine_rejected": len(genuine) - genuine_accepted,
        "impostor_total": len(impostor),
        "impostor_accepted": impostor_accepted,
        "impostor_rejected": len(impostor) - impostor_accepted,
        "correct": correct,
        "total": len(scored),
        "accuracy": correct / len(scored) if scored else None,
        "false_acceptances": impostor_accepted,
        "false_rejections": len(genuine) - genuine_accepted,
        "errors": len(rows) - len(scored),
        "genuine_scores": [r["similarity"] for r in genuine],
        "impostor_scores": [r["similarity"] for r in impostor],
    }


def format_table(rows):
    if not rows:
        return "(no recordings found)"

    width = max(len("Recording"), max(len(r["file"]) for r in rows))
    header = (f"{'Recording':<{width}}  {'Actual':<9}  {'Similarity':>10}  "
              f"{'Predicted':<10}  Result")
    lines = [header, "-" * len(header)]

    for r in rows:
        if r["error"]:
            lines.append(f"{r['file']:<{width}}  {r['label']:<9}  "
                         f"{'ERROR':>10}  {'-':<10}  {r['error']}")
            continue
        lines.append(
            f"{r['file']:<{width}}  {r['label']:<9}  "
            f"{r['similarity']:>10.4f}  {r['predicted']:<10}  "
            f"{'correct' if r['correct'] else 'INCORRECT'}"
        )
    return "\n".join(lines)


def _range_text(scores):
    if not scores:
        return "none"
    return (f"min {min(scores):.4f}, mean {sum(scores) / len(scores):.4f}, "
            f"max {max(scores):.4f}")


def format_summary(summary, threshold=SIMILARITY_THRESHOLD):
    s = summary
    lines = [
        f"Threshold: {threshold} (original value from the sample script; "
        f"not validated)",
        "",
        f"Genuine recordings : {s['genuine_total']:>3}  "
        f"accepted (TARGET) {s['genuine_accepted']}  |  "
        f"rejected (NON_TARGET) {s['genuine_rejected']}",
        f"Impostor recordings: {s['impostor_total']:>3}  "
        f"accepted (TARGET) {s['impostor_accepted']}  |  "
        f"rejected (NON_TARGET) {s['impostor_rejected']}",
    ]

    if s["accuracy"] is None:
        lines.append("Accuracy: n/a (no scored recordings)")
    else:
        lines.append(f"Accuracy: {s['correct']}/{s['total']} = "
                     f"{s['accuracy'] * 100:.1f}%")

    lines += [
        f"False acceptances (impostor accepted): {s['false_acceptances']}",
        f"False rejections  (genuine rejected) : {s['false_rejections']}",
    ]
    if s["errors"]:
        lines.append(f"Recordings that could not be scored: {s['errors']}")

    lines += [
        "",
        f"Genuine scores : {_range_text(s['genuine_scores'])}",
        f"Impostor scores: {_range_text(s['impostor_scores'])}",
    ]

    g, i = s["genuine_scores"], s["impostor_scores"]
    if g and i:
        if min(g) > max(i):
            lines.append(
                f"Separation: every genuine score ({min(g):.4f}+) is above "
                f"every impostor score (up to {max(i):.4f}) in this data, so "
                f"a threshold between those two values would separate it.")
        else:
            lines.append(
                f"Separation: the ranges OVERLAP (lowest genuine "
                f"{min(g):.4f}, highest impostor {max(i):.4f}); no single "
                f"threshold gets every recording right in this data.")

    notes = []
    if not i:
        notes.append("No impostor recordings: false acceptances cannot be "
                     "measured, and accuracy only reflects the genuine "
                     "recordings.")
    if not g:
        notes.append("No genuine recordings: false rejections cannot be "
                     "measured.")
    for name, count in (("genuine", s["genuine_total"]),
                        ("impostor", s["impostor_total"])):
        if 0 < count < MIN_PER_CLASS:
            notes.append(f"Only {count} {name} recordings (fewer than "
                         f"{MIN_PER_CLASS}): treat the result as an anecdote.")
    notes.append("These numbers describe only these recordings, this "
                 "microphone and these speakers. A similarity score is not "
                 "a probability, and this is not proof the system is reliable.")

    return "\n".join(lines + [""] + [f"NOTE: {n}" for n in notes])


def write_csv(rows, path=RESULTS_CSV):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "actual", "similarity", "predicted",
                         "correct", "error"])
        for r in rows:
            writer.writerow([
                r["file"], r["label"],
                "" if r["similarity"] is None else f"{r['similarity']:.6f}",
                r["predicted"], r["correct"] if r["correct"] is not None else "",
                r["error"],
            ])
    return path


def main():
    reference_hash = sha256_of(REFERENCE)

    try:
        recordings = find_recordings()
    except ValueError as error:
        print(error)
        return 2

    if not recordings:
        print(f"No recordings found under {RECORDINGS_DIR}\\genuine or "
              f"\\impostor.\nRecord some with: python -m "
              f"evaluation.record_session --help")
        return 2

    # Heavy imports only when scoring. The model is loaded ONCE.
    from audio import speaker_verification as sv

    model = sv.load_speaker_model()
    with contextlib.redirect_stdout(io.StringIO()):
        reference_embedding = sv.create_embedding(model, REFERENCE)

    def similarity_fn(path):
        with contextlib.redirect_stdout(io.StringIO()):   # mute "Prepared audio"
            embedding = sv.create_embedding(model, path)
        return sv.compare_embeddings(reference_embedding, embedding)

    rows = evaluate(recordings, similarity_fn)

    print()
    print(format_table(rows))
    print()
    print(format_summary(summarize(rows)))

    csv_path = write_csv(rows)
    print(f"\nResults saved to {csv_path}")

    unchanged = sha256_of(REFERENCE) == reference_hash
    print(f"reference.wav SHA-256 {reference_hash[:16]}... "
          f"{'unchanged' if unchanged else 'CHANGED - results invalid'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
