"""
Tests for the simple evaluation (evaluation/evaluate_speaker.py and
evaluation/record_session.py).

The similarity scores below are MADE-UP numbers. These tests check the
bookkeeping only: the TARGET/NON_TARGET rule, correct/incorrect, the counts
and the accuracy arithmetic. They are not evidence that ECAPA recognises
anyone.

Run from the project root:
    python -m unittest tests.test_simple_evaluation -v
"""

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io.wavfile import write

import audio.rolling_verifier as live
from evaluation import evaluate_speaker as ev
from evaluation import record_session as rec


def wav(path, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.default_rng(seed).integers(-5000, 5000, 16000)
    write(str(path), 16000, data.astype(np.int16))
    return path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TestRules(unittest.TestCase):

    def test_uses_the_live_threshold_unchanged(self):
        self.assertEqual(ev.SIMILARITY_THRESHOLD, 0.1)
        self.assertEqual(ev.SIMILARITY_THRESHOLD, live.SIMILARITY_THRESHOLD)

    def test_classify_is_inclusive_like_the_live_verifier(self):
        self.assertEqual(ev.classify(0.1), "TARGET")
        self.assertEqual(ev.classify(0.5), "TARGET")
        self.assertEqual(ev.classify(0.0999), "NON_TARGET")
        self.assertEqual(ev.classify(-0.2), "NON_TARGET")

    def test_classify_honours_a_given_threshold(self):
        self.assertEqual(ev.classify(0.3, threshold=0.4), "NON_TARGET")

    def test_correct_or_incorrect(self):
        self.assertTrue(ev.is_correct("genuine", "TARGET"))
        self.assertFalse(ev.is_correct("genuine", "NON_TARGET"))   # false reject
        self.assertTrue(ev.is_correct("impostor", "NON_TARGET"))
        self.assertFalse(ev.is_correct("impostor", "TARGET"))      # false accept


class TestEvaluateAndSummarize(unittest.TestCase):

    SCORES = {
        # genuine: 3 accepted, 1 rejected (false rejection)
        "g1.wav": 0.50, "g2.wav": 0.20, "g3.wav": 0.10, "g4.wav": 0.05,
        # impostor: 1 accepted (false acceptance), 3 rejected
        "i1.wav": 0.30, "i2.wav": 0.09, "i3.wav": 0.00, "i4.wav": -0.10,
    }

    def rows(self):
        recordings = [
            ("genuine" if name.startswith("g") else "impostor", Path(name))
            for name in self.SCORES
        ]
        return ev.evaluate(recordings, lambda p: self.SCORES[p.name])

    def test_row_values(self):
        by_name = {r["file"]: r for r in self.rows()}
        self.assertEqual(by_name["g1.wav"]["predicted"], "TARGET")
        self.assertTrue(by_name["g1.wav"]["correct"])
        self.assertEqual(by_name["g3.wav"]["predicted"], "TARGET")   # == 0.1
        self.assertEqual(by_name["g4.wav"]["predicted"], "NON_TARGET")
        self.assertFalse(by_name["g4.wav"]["correct"])
        self.assertEqual(by_name["i1.wav"]["predicted"], "TARGET")
        self.assertFalse(by_name["i1.wav"]["correct"])
        self.assertTrue(by_name["i2.wav"]["correct"])

    def test_summary_counts(self):
        s = ev.summarize(self.rows())
        self.assertEqual((s["genuine_total"], s["genuine_accepted"],
                          s["genuine_rejected"]), (4, 3, 1))
        self.assertEqual((s["impostor_total"], s["impostor_accepted"],
                          s["impostor_rejected"]), (4, 1, 3))
        self.assertEqual(s["false_rejections"], 1)
        self.assertEqual(s["false_acceptances"], 1)
        self.assertEqual((s["correct"], s["total"]), (6, 8))
        self.assertAlmostEqual(s["accuracy"], 0.75)

    def test_perfect_and_worst_cases(self):
        perfect = ev.evaluate(
            [("genuine", Path("a")), ("impostor", Path("b"))],
            lambda p: 0.5 if p.name == "a" else 0.0)
        self.assertEqual(ev.summarize(perfect)["accuracy"], 1.0)

        worst = ev.evaluate(
            [("genuine", Path("a")), ("impostor", Path("b"))],
            lambda p: 0.0 if p.name == "a" else 0.5)
        s = ev.summarize(worst)
        self.assertEqual(s["accuracy"], 0.0)
        self.assertEqual((s["false_acceptances"], s["false_rejections"]), (1, 1))

    def test_unscorable_file_is_reported_and_not_counted(self):
        def similarity(path):
            if path.name == "bad.wav":
                raise RuntimeError("cannot read")
            return 0.5

        rows = ev.evaluate(
            [("genuine", Path("ok.wav")), ("impostor", Path("bad.wav"))],
            similarity)
        s = ev.summarize(rows)
        self.assertEqual(s["errors"], 1)
        self.assertEqual(s["total"], 1)
        self.assertEqual(s["impostor_total"], 0)
        self.assertIn("ERROR", ev.format_table(rows))

    def test_empty_input(self):
        s = ev.summarize([])
        self.assertIsNone(s["accuracy"])
        self.assertEqual(ev.format_table([]), "(no recordings found)")


class TestOutputText(unittest.TestCase):

    def summary_for(self, genuine, impostor):
        recordings = (
            [("genuine", Path(f"g{i}")) for i in range(len(genuine))]
            + [("impostor", Path(f"i{i}")) for i in range(len(impostor))]
        )
        scores = {f"g{i}": v for i, v in enumerate(genuine)}
        scores.update({f"i{i}": v for i, v in enumerate(impostor)})
        rows = ev.evaluate(recordings, lambda p: scores[p.name])
        return ev.format_summary(ev.summarize(rows)), rows

    def test_table_has_all_requested_columns(self):
        _, rows = self.summary_for([0.4], [0.05])
        table = ev.format_table(rows)
        for text in ("Recording", "Actual", "Similarity", "Predicted",
                     "genuine", "impostor", "0.4000", "TARGET",
                     "NON_TARGET", "correct"):
            self.assertIn(text, table)

    def test_summary_has_all_requested_lines(self):
        text, _ = self.summary_for([0.4, 0.05], [0.3, 0.0])
        for part in ("Genuine recordings", "Impostor recordings", "accepted",
                     "rejected", "Accuracy: 2/4 = 50.0%",
                     "False acceptances", "False rejections", "Threshold: 0.1"):
            self.assertIn(part, text)

    def test_summary_never_shows_scores_as_percentages(self):
        text, _ = self.summary_for([0.4], [0.05])
        # The only percent sign is the accuracy line.
        self.assertEqual(text.count("%"), 1)
        self.assertIn("not a probability", text)

    def test_overlap_and_separation_messages(self):
        overlapping, _ = self.summary_for([0.2, 0.05], [0.3, 0.0])
        self.assertIn("OVERLAP", overlapping)

        separated, _ = self.summary_for([0.4, 0.3], [0.05, 0.0])
        self.assertIn("would separate it", separated)

    def test_missing_class_and_small_sample_warnings(self):
        text, _ = self.summary_for([0.4, 0.5], [])
        self.assertIn("No impostor recordings", text)
        self.assertIn("anecdote", text)

        text, _ = self.summary_for([], [0.1])
        self.assertIn("No genuine recordings", text)


class TestFindRecordings(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.reference = wav(self.root / "reference.wav", 0)
        self.dir = self.root / "recordings"

    def test_label_comes_from_the_folder_including_subfolders(self):
        wav(self.dir / "genuine" / "me" / "S1" / "a.wav", 1)
        wav(self.dir / "genuine" / "b.wav", 2)
        wav(self.dir / "impostor" / "spk01" / "c.wav", 3)
        (self.dir / "impostor" / "notes.txt").write_text("ignored")

        found = ev.find_recordings(self.dir, self.reference)

        self.assertEqual(
            sorted((label, p.name) for label, p in found),
            [("genuine", "a.wav"), ("genuine", "b.wav"), ("impostor", "c.wav")])

    def test_copy_of_reference_is_refused(self):
        copy = self.dir / "genuine" / "me" / "copy.wav"
        copy.parent.mkdir(parents=True)
        copy.write_bytes(self.reference.read_bytes())
        with self.assertRaises(ValueError) as ctx:
            ev.find_recordings(self.dir, self.reference)
        self.assertIn("identical to reference.wav", str(ctx.exception))

    def test_missing_folders_give_empty_list(self):
        self.assertEqual(ev.find_recordings(self.dir, self.reference), [])

    def test_evaluation_never_changes_reference_or_recordings(self):
        a = wav(self.dir / "genuine" / "a.wav", 1)
        b = wav(self.dir / "impostor" / "b.wav", 2)
        before = (sha(self.reference), sha(a), sha(b))

        found = ev.find_recordings(self.dir, self.reference)
        rows = ev.evaluate(found, lambda p: 0.5)
        ev.write_csv(rows, self.root / "out" / "results.csv")

        self.assertEqual(before, (sha(self.reference), sha(a), sha(b)))
        text = (self.root / "out" / "results.csv").read_text()
        self.assertIn("file,actual,similarity,predicted,correct,error", text)
        self.assertIn("a.wav,genuine,0.500000,TARGET,True", text)


class TestRecorder(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.latest = self.root / "latest.wav"
        self.calls = 0

    def capture(self):
        self.calls += 1
        if self.calls == 1:
            return None                       # simulate "no speech"
        wav(self.latest, 100 + self.calls)
        return self.latest

    def record(self, label, name, count):
        return rec.record_samples(
            label, name, count, capture_fn=self.capture,
            info={"full_path": str(self.latest)},
            recordings_dir=self.root / "recordings", log=lambda *_: None)

    def test_saves_numbered_files_in_label_folder(self):
        self.assertEqual(self.record("impostor", "spk01", 2), 2)
        names = sorted(p.name for p in
                       (self.root / "recordings" / "impostor" / "spk01").glob("*.wav"))
        self.assertEqual(names, ["impostor_spk01_001.wav",
                                 "impostor_spk01_002.wav"])

    def test_never_overwrites_existing_files(self):
        folder = self.root / "recordings" / "genuine" / "me"
        existing = wav(folder / "genuine_me_001.wav", 1)
        before = sha(existing)

        self.record("genuine", "me", 1)

        self.assertEqual(sha(existing), before)
        self.assertTrue((folder / "genuine_me_002.wav").is_file())

    def test_numbering_continues_across_subfolders(self):
        wav(self.root / "recordings" / "genuine" / "me" / "S1"
            / "genuine_me_001.wav", 1)
        path = rec.next_path(self.root / "recordings" / "genuine" / "me",
                             "genuine", "me")
        self.assertEqual(path.name, "genuine_me_002.wav")


if __name__ == "__main__":
    unittest.main()
