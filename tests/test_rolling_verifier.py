"""
Unit tests for audio/rolling_verifier.py.

All audio here is synthetic and a fake embedding function stands in for
ECAPA. These tests check BUFFER, TIMING and STATE mechanics only. They
say nothing about whether ECAPA identifies a speaker correctly.

Run from the project root:
    python -m unittest tests.test_rolling_verifier -v
"""

import threading
import unittest

import numpy as np
import torch

from audio.rolling_verifier import (
    RollingSpeakerVerifier,
    VerificationState,
    SIMILARITY_THRESHOLD,
)

RATE = 16000
CHUNK = 1600                       # 100 ms, like the microphone blocks
REFERENCE = torch.tensor([[1.0, 0.0]])


def chunk(value=0.0, n=CHUNK):
    return np.full(n, value, dtype=np.float32)


def make_embed(similarity=0.5, calls=None, windows=None):
    """Fake embed_fn whose cosine with REFERENCE equals `similarity`."""

    def embed(audio):
        if calls is not None:
            calls.append(len(audio))
        if windows is not None:
            windows.append(audio.copy())
        s = similarity() if callable(similarity) else similarity
        return torch.tensor([[s, (1 - s * s) ** 0.5]])

    return embed


def make_verifier(embed_fn=None, **kwargs):
    return RollingSpeakerVerifier(
        reference_embedding=REFERENCE,
        embed_fn=embed_fn or make_embed(),
        **kwargs,
    )


def feed_seconds(verifier, seconds, value=0.0):
    results = []
    for _ in range(int(round(seconds * 10))):
        r = verifier.feed(chunk(value))
        if r is not None:
            results.append(r)
    return results


class TestInsufficientAudio(unittest.TestCase):

    def test_less_than_two_seconds_is_insufficient_not_non_target(self):
        calls = []
        v = make_verifier(make_embed(0.0, calls))   # would be NON_TARGET

        results = feed_seconds(v, 1.9)

        self.assertTrue(results)
        for r in results:
            self.assertEqual(r.state, VerificationState.INSUFFICIENT_AUDIO)
            self.assertIsNone(r.similarity)
            self.assertIsNone(r.inference_ms)
        self.assertEqual(calls, [])                 # no inference run

    def test_short_answer_like_yes_is_never_non_target(self):
        v = make_verifier(make_embed(-1.0))
        results = feed_seconds(v, 0.8)              # ~ "yes"
        self.assertTrue(all(
            r.state is VerificationState.INSUFFICIENT_AUDIO for r in results
        ))

    def test_exactly_two_seconds_runs_inference(self):
        calls = []
        v = make_verifier(make_embed(0.5, calls))

        results = feed_seconds(v, 2.0)

        self.assertEqual(results[-1].state, VerificationState.TARGET)
        self.assertEqual(results[-1].buffered_sec, 2.0)
        self.assertEqual(calls, [2 * RATE])

    def test_just_below_two_seconds_is_still_insufficient(self):
        calls = []
        v = make_verifier(make_embed(0.5, calls))
        r = v.feed(chunk(n=2 * RATE - 1))           # one sample short
        self.assertEqual(r.state, VerificationState.INSUFFICIENT_AUDIO)
        self.assertEqual(calls, [])


class TestRollingBuffer(unittest.TestCase):

    def test_buffer_never_exceeds_window(self):
        v = make_verifier()
        feed_seconds(v, 5.0)
        self.assertEqual(v.buffered_sec, 2.0)

    def test_window_contains_only_most_recent_audio(self):
        windows = []
        v = make_verifier(make_embed(0.5, windows=windows))

        feed_seconds(v, 2.0, value=0.1)   # old audio
        feed_seconds(v, 2.0, value=0.9)   # new audio replaces it

        last = windows[-1]
        self.assertEqual(len(last), 2 * RATE)
        self.assertTrue(np.allclose(last, 0.9))

    def test_window_slides_part_way(self):
        windows = []
        v = make_verifier(make_embed(0.5, windows=windows))

        feed_seconds(v, 2.0, value=0.1)
        feed_seconds(v, 0.5, value=0.9)   # one more step

        last = windows[-1]
        self.assertTrue(np.allclose(last[: int(1.5 * RATE)], 0.1))
        self.assertTrue(np.allclose(last[int(1.5 * RATE):], 0.9))

    def test_int16_input_is_scaled_to_float(self):
        windows = []
        v = make_verifier(make_embed(0.5, windows=windows))
        for _ in range(20):
            v.feed(np.full(CHUNK, 16384, dtype=np.int16))
        self.assertAlmostEqual(float(windows[-1][0]), 0.5, places=4)

    def test_reset_clears_buffer(self):
        v = make_verifier()
        feed_seconds(v, 2.0)
        v.reset()
        self.assertEqual(v.buffered_sec, 0.0)


class TestVerificationInterval(unittest.TestCase):

    def test_one_tick_every_500ms(self):
        v = make_verifier()
        results = feed_seconds(v, 3.0)
        self.assertEqual(len(results), 6)           # 3.0 s / 0.5 s
        times = [r.total_sec for r in results]
        self.assertEqual(times, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    def test_inference_runs_every_step_once_buffer_is_full(self):
        calls = []
        v = make_verifier(make_embed(0.5, calls))
        feed_seconds(v, 3.0)
        self.assertEqual(len(calls), 3)             # ticks at 2.0, 2.5, 3.0

    def test_small_chunks_accumulate_to_a_tick(self):
        v = make_verifier()
        for _ in range(9):                          # 9 x 50 ms = 0.45 s
            self.assertIsNone(v.feed(chunk(n=800)))
        self.assertIsNotNone(v.feed(chunk(n=800)))  # 10th reaches 0.5 s

    def test_large_chunk_causes_single_tick_not_backlog(self):
        calls = []
        v = make_verifier(make_embed(0.5, calls))
        v.feed(chunk(n=10 * RATE))
        self.assertEqual(len(calls), 1)


class TestSimilarityAndState(unittest.TestCase):

    def test_default_threshold_is_original_value(self):
        self.assertEqual(SIMILARITY_THRESHOLD, 0.1)
        self.assertEqual(make_verifier().threshold, 0.1)

    def test_target_when_similarity_above_threshold(self):
        v = make_verifier(make_embed(0.35))
        r = feed_seconds(v, 2.0)[-1]
        self.assertEqual(r.state, VerificationState.TARGET)
        self.assertAlmostEqual(r.similarity, 0.35, places=5)

    def test_non_target_when_similarity_below_threshold(self):
        v = make_verifier(make_embed(0.05))
        r = feed_seconds(v, 2.0)[-1]
        self.assertEqual(r.state, VerificationState.NON_TARGET)
        self.assertAlmostEqual(r.similarity, 0.05, places=5)

    def test_threshold_boundary_is_inclusive(self):
        v = make_verifier(make_embed(0.25), threshold=0.25)
        r = feed_seconds(v, 2.0)[-1]
        self.assertEqual(r.state, VerificationState.TARGET)

    def test_threshold_is_configurable(self):
        v = make_verifier(make_embed(0.35), threshold=0.6)
        r = feed_seconds(v, 2.0)[-1]
        self.assertEqual(r.state, VerificationState.NON_TARGET)

    def test_inference_duration_measured_with_injected_clock(self):
        ticks = iter([10.0, 10.25])                 # start, end -> 250 ms
        v = make_verifier(clock=lambda: next(ticks))
        r = feed_seconds(v, 2.0)[-1]
        self.assertAlmostEqual(r.inference_ms, 250.0, places=3)

    def test_log_only_audio_is_not_modified_or_dropped(self):
        windows = []
        v = make_verifier(make_embed(0.0, windows=windows))   # NON_TARGET
        feed_seconds(v, 3.0, value=0.3)
        # Every non-target window still holds the full audio untouched.
        self.assertTrue(all(np.allclose(w, 0.3) for w in windows))
        self.assertEqual(v.buffered_sec, 2.0)

    def test_on_result_callback_and_latest_result(self):
        seen = []
        v = make_verifier(on_result=seen.append)
        feed_seconds(v, 2.0)
        self.assertEqual(len(seen), 4)
        self.assertIs(v.latest_result, seen[-1])

    def test_embedding_error_is_recorded_not_raised(self):
        def boom(_):
            raise RuntimeError("bad audio")

        v = make_verifier(boom)
        feed_seconds(v, 2.0)                        # must not raise
        self.assertEqual(v.error_count, 1)
        self.assertIsInstance(v.last_error, RuntimeError)
        self.assertFalse(v.is_busy)


class TestInferenceAlreadyRunning(unittest.TestCase):

    def test_tick_skipped_while_inference_running(self):
        started = threading.Event()
        release = threading.Event()
        calls = []

        def slow_embed(audio):
            calls.append(len(audio))
            started.set()
            release.wait(timeout=5)
            return torch.tensor([[0.5, 0.75 ** 0.5]])

        v = make_verifier(slow_embed, background=True)

        feed_seconds(v, 2.0)                        # first inference starts
        self.assertTrue(started.wait(timeout=2))
        self.assertTrue(v.is_busy)

        feed_seconds(v, 1.5)                        # 3 ticks while busy
        self.assertEqual(v.skipped_ticks, 3)
        self.assertEqual(len(calls), 1)             # nothing was queued

        release.set()
        v.wait_idle(timeout=5)

        self.assertFalse(v.is_busy)
        self.assertEqual(v.latest_result.state, VerificationState.TARGET)

        # After it finishes, the next tick runs normally again.
        feed_seconds(v, 0.5)
        v.wait_idle(timeout=5)
        self.assertEqual(len(calls), 2)

    def test_no_backlog_after_slow_inference_completes(self):
        release = threading.Event()
        calls = []

        def slow_embed(audio):
            calls.append(1)
            release.wait(timeout=5)
            return torch.tensor([[0.5, 0.75 ** 0.5]])

        v = make_verifier(slow_embed, background=True)
        feed_seconds(v, 4.0)                        # 1 run + 3 skipped
        release.set()
        v.wait_idle(timeout=5)
        self.assertEqual(len(calls), 1)

    def test_insufficient_audio_still_reported_while_busy_is_not_needed(self):
        # Busy implies a full window existed, so a skipped tick can only
        # happen after enough audio; the skip never hides INSUFFICIENT.
        v = make_verifier(background=True)
        results = feed_seconds(v, 1.5)
        self.assertTrue(all(
            r.state is VerificationState.INSUFFICIENT_AUDIO for r in results
        ))
        self.assertEqual(v.skipped_ticks, 0)

    def test_background_result_delivered_via_callback(self):
        seen = []
        done = threading.Event()

        def cb(r):
            seen.append(r)
            if r.state is not VerificationState.INSUFFICIENT_AUDIO:
                done.set()

        v = make_verifier(background=True, on_result=cb)
        feed_seconds(v, 2.0)
        self.assertTrue(done.wait(timeout=2))
        self.assertEqual(seen[-1].state, VerificationState.TARGET)


class TestTurnsStatsAndSafety(unittest.TestCase):

    def test_require_turn_ignores_audio_until_begin_turn(self):
        calls = []
        v = make_verifier(make_embed(0.5, calls), require_turn=True)

        self.assertEqual(feed_seconds(v, 3.0), [])
        self.assertEqual(v.ignored_chunks, 30)
        self.assertEqual(v.buffered_sec, 0.0)

        v.begin_turn()
        self.assertTrue(feed_seconds(v, 2.0))

    def test_end_turn_returns_stats_and_stops_accepting(self):
        v = make_verifier(require_turn=True)
        v.begin_turn()
        feed_seconds(v, 3.0)

        stats = v.end_turn()

        self.assertEqual(stats.ticks, 6)
        self.assertEqual(stats.insufficient, 3)
        self.assertEqual(stats.target, 3)
        self.assertEqual(stats.inferences, 3)

        feed_seconds(v, 1.0)
        self.assertEqual(v.stats.ticks, 6)          # nothing added

    def test_skip_callback_and_skip_count(self):
        release = threading.Event()
        skipped_at = []

        def slow(audio):
            release.wait(timeout=5)
            return torch.tensor([[0.5, 0.75 ** 0.5]])

        v = make_verifier(slow, background=True, on_skip=skipped_at.append)
        feed_seconds(v, 3.0)                        # 1 run, 2 skipped
        release.set()
        v.wait_idle(timeout=5)

        self.assertEqual(skipped_at, [2.5, 3.0])
        self.assertEqual(v.stats.skipped, 2)

    def test_worker_gets_a_snapshot_not_the_live_buffer(self):
        release = threading.Event()
        seen = []

        def slow(audio):
            release.wait(timeout=5)
            seen.append(audio.copy())               # read AFTER later feeds
            return torch.tensor([[0.5, 0.75 ** 0.5]])

        v = make_verifier(slow, background=True)
        feed_seconds(v, 2.0, value=0.1)
        feed_seconds(v, 1.0, value=0.9)             # buffer changes meanwhile
        release.set()
        v.wait_idle(timeout=5)

        self.assertTrue(np.allclose(seen[0], 0.1))

    def test_stale_result_from_previous_turn_is_discarded(self):
        release = threading.Event()

        def slow(audio):
            release.wait(timeout=5)
            return torch.tensor([[0.5, 0.75 ** 0.5]])

        v = make_verifier(slow, background=True)
        feed_seconds(v, 2.0)
        self.assertTrue(v.is_busy)

        v.reset(timeout=0.05)                       # gives up waiting
        release.set()
        v.wait_idle(timeout=5)

        self.assertEqual(v.stats.target, 0)
        self.assertIsNone(v.latest_result)

    def test_no_duplicate_inference_for_one_tick(self):
        calls = []
        v = make_verifier(make_embed(0.5, calls), background=True)
        feed_seconds(v, 2.0)
        v.wait_idle(timeout=5)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
