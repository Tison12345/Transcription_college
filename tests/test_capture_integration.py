"""
Integration tests: RollingSpeakerVerifier inside capture_one_utterance().

A fake sounddevice InputStream replays deterministic 100 ms int16 chunks
through the real capture loop, and the speech probability is patched
(1.0 for loud frames, 0.0 for silent ones). A fake embedding function
replaces ECAPA.

These tests check audio flow, threading, reset and statistics. The audio
is synthetic, so they say nothing about speaker-identity accuracy.

Run from the project root:
    python -m unittest tests.test_capture_integration -v
"""

import contextlib
import io
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from scipy.io.wavfile import read

import audio.utterance_detector as ud
from audio.rolling_verifier import (
    RollingSpeakerVerifier,
    format_result,
    format_skip,
)

REFERENCE = torch.tensor([[1.0, 0.0]])
CHUNK = 1600                    # 100 ms, same as the real microphone block
LOUD = 8000


def make_stream_audio(loud=LOUD, lead=0.3, speech=3.0, tail=2.0):
    """int16 audio: silence, 'speech', then endpoint silence."""

    parts = [
        np.zeros(int(lead * 16000), dtype=np.int16),
        np.full(int(speech * 16000), loud, dtype=np.int16),
        np.zeros(int(tail * 16000), dtype=np.int16),
    ]
    return np.concatenate(parts)


def fake_input_stream(audio):
    """Build a stand-in for sd.InputStream that replays `audio`."""

    state = {"open": False, "opened": 0}

    class FakeInputStream:
        def __init__(self, **kwargs):
            self._callback = kwargs["callback"]

        def __enter__(self):
            state["open"] = True
            state["opened"] += 1
            for i in range(0, len(audio), CHUNK):
                block = audio[i:i + CHUNK].reshape(-1, 1)
                self._callback(block, len(block), None, None)
            return self

        def __exit__(self, *exc):
            state["open"] = False
            return False

    return FakeInputStream, state


def fake_probability(frame):
    return 1.0 if np.max(np.abs(frame)) > 1000 else 0.0


class CaptureTestCase(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)

        patches = [
            mock.patch.object(ud, "get_speech_probability", fake_probability),
            mock.patch.object(ud, "LATEST_FULL_AUDIO", tmp / "full.wav"),
            mock.patch.object(ud, "LATEST_SPEAKER_AUDIO", tmp / "speaker.wav"),
            mock.patch.object(ud, "HISTORY_DIRECTORY", tmp / "history"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        self.tmp = tmp

    def capture(self, audio, verifier=None):
        stream_cls, state = fake_input_stream(audio)
        out = io.StringIO()
        with mock.patch.object(ud.sd, "InputStream", stream_cls):
            with contextlib.redirect_stdout(out):
                path = ud.capture_one_utterance(verifier=verifier)
        return path, out.getvalue(), state

    def make_verifier(self, similarity=0.5, delay=0.0, windows=None,
                      concurrency=None, background=True):
        lock = threading.Lock()
        calls = []

        def embed(audio):
            with lock:
                calls.append(len(audio))
                if concurrency is not None:
                    concurrency["now"] += 1
                    concurrency["max"] = max(
                        concurrency["max"], concurrency["now"]
                    )
            if windows is not None:
                windows.append(audio.copy())
            if delay:
                time.sleep(delay)
            if concurrency is not None:
                with lock:
                    concurrency["now"] -= 1
            return torch.tensor([[similarity, (1 - similarity ** 2) ** 0.5]])

        verifier = RollingSpeakerVerifier(
            reference_embedding=REFERENCE,
            embed_fn=embed,
            background=background,
            require_turn=True,
            # Same callbacks main.py installs.
            on_result=lambda r: print(format_result(r)),
            on_skip=lambda t: print(format_skip(t)),
        )
        return verifier, calls


class TestLogOnlyNoAudioLoss(CaptureTestCase):

    def test_saved_audio_identical_with_and_without_verifier(self):
        audio = make_stream_audio()

        path_a, _, _ = self.capture(audio, verifier=None)
        full_a = read(str(path_a))[1].copy()
        speaker_a = read(str(ud.LATEST_SPEAKER_AUDIO))[1].copy()

        # A verifier that says NON_TARGET for everything.
        verifier, _ = self.make_verifier(similarity=-1.0)
        path_b, _, _ = self.capture(audio, verifier=verifier)
        full_b = read(str(path_b))[1]
        speaker_b = read(str(ud.LATEST_SPEAKER_AUDIO))[1]

        np.testing.assert_array_equal(full_a, full_b)
        np.testing.assert_array_equal(speaker_a, speaker_b)

    def test_verifier_receives_exactly_the_recorded_audio(self):
        verifier, _ = self.make_verifier()
        fed = []
        real_feed = verifier.feed
        verifier.feed = lambda chunk: (fed.append(np.array(chunk)),
                                       real_feed(chunk))[1]

        path, _, _ = self.capture(make_stream_audio(), verifier)

        recorded = read(str(path))[1]
        np.testing.assert_array_equal(np.concatenate(fed), recorded)

    def test_utterance_still_returned_for_non_target_audio(self):
        verifier, _ = self.make_verifier(similarity=-1.0)
        path, _, _ = self.capture(make_stream_audio(), verifier)
        self.assertIsNotNone(path)
        self.assertIn("rolling", ud.LAST_CAPTURE_INFO)
        self.assertEqual(ud.LAST_CAPTURE_INFO["rolling"]["target"], 0)

    def test_failing_verifier_does_not_break_capture(self):
        class Broken:
            def begin_turn(self): pass
            def feed(self, _): raise RuntimeError("boom")
            def end_turn(self):
                from audio.rolling_verifier import VerificationStats
                return VerificationStats()

        path, out, _ = self.capture(make_stream_audio(), Broken())
        self.assertIsNotNone(path)
        self.assertIn("verifier error", out)


class TestResetBetweenUtterances(CaptureTestCase):

    def test_second_turn_has_no_audio_from_first_turn(self):
        windows = []
        verifier, _ = self.make_verifier(windows=windows)

        self.capture(make_stream_audio(loud=8000), verifier)
        first_turn_value = np.float32(8000 / 32768.0)
        self.assertTrue(windows)

        windows.clear()
        self.capture(make_stream_audio(loud=16000), verifier)

        self.assertTrue(windows)
        for w in windows:
            self.assertFalse(np.any(np.isclose(w, first_turn_value)))

    def test_statistics_start_fresh_each_turn(self):
        verifier, _ = self.make_verifier(background=False)

        self.capture(make_stream_audio(), verifier)
        first = dict(ud.LAST_CAPTURE_INFO["rolling"])

        self.capture(make_stream_audio(), verifier)
        second = dict(ud.LAST_CAPTURE_INFO["rolling"])

        self.assertEqual(first, second)         # not doubled


class TestBackgroundWorker(CaptureTestCase):

    def test_worker_finishes_before_summary_and_results_counted(self):
        verifier, calls = self.make_verifier(delay=0.05)
        _, out, _ = self.capture(make_stream_audio(), verifier)

        self.assertFalse(verifier.is_busy)
        stats = ud.LAST_CAPTURE_INFO["rolling"]
        self.assertEqual(stats["inferences"], len(calls))
        self.assertIn("utterance summary", out)

    def test_ticks_skipped_while_inference_busy_and_never_overlap(self):
        concurrency = {"now": 0, "max": 0}
        verifier, calls = self.make_verifier(
            delay=0.3, concurrency=concurrency
        )

        _, out, _ = self.capture(make_stream_audio(), verifier)
        stats = ud.LAST_CAPTURE_INFO["rolling"]

        # The fake stream is replayed faster than real time, so a slow
        # inference must cause skipped ticks, and never two at once.
        self.assertGreater(stats["skipped"], 0)
        self.assertEqual(concurrency["max"], 1)
        self.assertEqual(len(calls), stats["inferences"])
        self.assertIn("tick SKIPPED", out)

    def test_statistics_add_up(self):
        verifier, _ = self.make_verifier(similarity=0.0, delay=0.02)
        self.capture(make_stream_audio(), verifier)
        s = ud.LAST_CAPTURE_INFO["rolling"]

        self.assertEqual(
            s["ticks"],
            s["target"] + s["non_target"] + s["insufficient"]
            + s["skipped"] + s["errors"],
        )
        self.assertEqual(s["target"], 0)        # similarity 0.0 < 0.1
        self.assertGreaterEqual(s["insufficient"], 1)
        self.assertEqual(s["errors"], 0)

    def test_exact_statistics_in_synchronous_mode(self):
        verifier, calls = self.make_verifier(background=False)
        self.capture(make_stream_audio(), verifier)
        s = ud.LAST_CAPTURE_INFO["rolling"]

        self.assertEqual(s["skipped"], 0)
        self.assertEqual(s["inferences"], len(calls))
        self.assertEqual(s["target"], len(calls))   # similarity 0.5 >= 0.1
        self.assertTrue(all(n == 2 * 16000 for n in calls))

    def test_log_lines_contain_state_similarity_and_inference_time(self):
        verifier, _ = self.make_verifier(background=False)

        _, out, _ = self.capture(make_stream_audio(), verifier)

        joined = "\n".join(
            line for line in out.splitlines() if line.startswith("[rolling]")
        )
        self.assertIn("INSUFFICIENT_AUDIO", joined)
        self.assertIn("TARGET", joined)
        self.assertIn("sim=0.5000", joined)
        self.assertRegex(joined, r"infer=\d+ ms")
        self.assertNotIn("%", joined)               # never a percentage


class TestTurnBoundaryAndTTS(CaptureTestCase):

    def test_audio_outside_a_turn_is_ignored(self):
        verifier, calls = self.make_verifier(background=False)

        # e.g. assistant TTS time: no turn is active.
        for _ in range(60):
            verifier.feed(np.full(CHUNK, LOUD, dtype=np.int16))

        self.assertEqual(verifier.ignored_chunks, 60)
        self.assertEqual(verifier.buffered_sec, 0.0)
        self.assertEqual(calls, [])

    def test_verifier_stops_accepting_after_capture_ends(self):
        verifier, _ = self.make_verifier(background=False)
        self.capture(make_stream_audio(), verifier)

        verifier.feed(np.full(CHUNK, LOUD, dtype=np.int16))
        self.assertEqual(verifier.ignored_chunks, 1)

    def test_no_speech_returns_none_and_prints_no_summary(self):
        verifier, calls = self.make_verifier(background=False)
        silence = np.zeros(16000, dtype=np.int16)

        with mock.patch.object(ud, "CAPTURE_TIMEOUT", 0.3):
            stream_cls, _ = fake_input_stream(silence)
            out = io.StringIO()
            with mock.patch.object(ud.sd, "InputStream", stream_cls):
                with contextlib.redirect_stdout(out):
                    path = ud.capture_one_utterance(
                        timeout=0.3, verifier=verifier
                    )

        self.assertIsNone(path)
        self.assertEqual(calls, [])
        self.assertNotIn("utterance summary", out.getvalue())


if __name__ == "__main__":
    unittest.main()
