"""
Decide how much to trust a Whisper transcript, from the metrics Whisper
already returns (avg_logprob, avg_no_speech_prob).

THRESHOLDS ARE NOT CALIBRATED FOR THIS PROJECT.

  * UNRELIABLE values are Whisper's own decoding defaults
    (log_prob_threshold = -1.0, no_speech_threshold = 0.6): Whisper itself
    treats a segment below -1.0 average log-probability, or above 0.6
    no-speech probability, as a failed or empty decode.
  * BORDERLINE values are placeholders chosen between "typical good" and
    "unreliable" in the few turns seen so far (good turns: logprob -0.35 to
    -0.65, no-speech 0.05 to 0.23). They need tuning on recorded calls.

A confident transcript can still be wrong ("my food hurts" scored about
-0.54 / 0.23 and would pass). These metrics only catch transcripts Whisper
itself was unsure about.
"""

OK = "ok"
BORDERLINE = "borderline"
UNRELIABLE = "unreliable"

# Whisper's own defaults.
UNRELIABLE_LOGPROB = -1.0
UNRELIABLE_NO_SPEECH = 0.6

# Placeholders - tune on real recordings.
BORDERLINE_LOGPROB = -0.8
BORDERLINE_NO_SPEECH = 0.4


def assess_transcript(avg_logprob, avg_no_speech_prob, short_valid_answer=False):
    """
    Return OK, BORDERLINE or UNRELIABLE.

    Missing metrics (None) are not held against the transcript.

    short_valid_answer: the text is a bare yes/no-style answer. Such short
    replies naturally get poor average log-probabilities, so only a high
    no-speech probability (probably noise) can reject them.
    """

    no_speech = avg_no_speech_prob
    logprob = avg_logprob

    if no_speech is not None and no_speech > UNRELIABLE_NO_SPEECH:
        return UNRELIABLE

    if short_valid_answer:
        return OK

    if logprob is not None and logprob < UNRELIABLE_LOGPROB:
        return UNRELIABLE

    if logprob is not None and logprob < BORDERLINE_LOGPROB:
        return BORDERLINE

    if no_speech is not None and no_speech > BORDERLINE_NO_SPEECH:
        return BORDERLINE

    return OK
