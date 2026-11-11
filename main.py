
import traceback

import sounddevice as sd

from audio.utterance_detector import (
    capture_one_utterance,
    LAST_CAPTURE_INFO,
)
from audio.transcribe import (
    load_whisper_model,
    transcribe_audio_with_confidence,
)
from audio.speaker_verification import (
    load_speaker_model,
    create_embedding,
    verify_speaker,
)
from audio.rolling_verifier import (
    RollingSpeakerVerifier,
    format_result,
    format_skip,
)
from dialogue.call_loop import run_conversation
from dialogue.state_machine import DialogueStateMachine


REFERENCE_AUDIO = "reference.wav"



def listen(
    whisper_model,
    speaker_model,
    reference_embedding,
    rolling_verifier=None,
):
    """Capture speech, compare the voice, and transcribe it."""

    # The rolling verifier is log-only: it never changes the audio.
    audio_file = capture_one_utterance(verifier=rolling_verifier)

    if audio_file is None:
        return None

    # Speaker verification
    print("\nChecking speaker similarity...")

    # ECAPA uses the trimmed recording; Whisper below uses the full one.
    speaker_file = LAST_CAPTURE_INFO.get("speaker_path", audio_file)

    # Log-only feature: a failure here must never end the call.
    try:
        similarity = verify_speaker(
            speaker_model,
            reference_embedding,
            speaker_file
        )
        print(f"Speaker similarity score: {similarity:.4f}")
    except Exception:
        similarity = None
        print("[error] speaker check failed (ignored, log-only):")
        traceback.print_exc()

    print("Speaker verification is currently running in monitoring mode.")

    # Speech recognition
    print("\nTranscribing...")

    result = transcribe_audio_with_confidence(
        whisper_model,
        audio_file
    )

    transcript = result["text"].strip()

    if not any(ch.isalnum() for ch in transcript):
        return None

    result["speaker_similarity"] = similarity

    return result


def main():

    print("\nLoading Whisper model...")
    whisper_model = load_whisper_model()

    print("\nLoading speaker verification model...")
    speaker_model = load_speaker_model()

    print("\nCreating enrolled patient voice reference...")
    reference_embedding = create_embedding(
        speaker_model,
        REFERENCE_AUDIO
    )

    print("Patient reference voice loaded successfully!")

    # Created once: reuses the loaded ECAPA model and the reference
    # embedding. Log-only - results are printed, nothing is filtered.
    # background=True keeps ECAPA inference off the capture loop.
    rolling_verifier = RollingSpeakerVerifier(
        model=speaker_model,
        reference_embedding=reference_embedding,
        background=True,
        require_turn=True,
        on_result=lambda result: print(format_result(result)),
        on_skip=lambda total_sec: print(format_skip(total_sec)),
    )

    agent = DialogueStateMachine()

    print("\nPatient follow-up assistant is ready!")

    def report(result):
        print(f"\nPatient: {result['text']}")

        print("\n--- Speaker Verification ---")
        similarity = result.get("speaker_similarity")
        if similarity is None:
            print("Similarity Score: unavailable")
        else:
            print(f"Similarity Score: {similarity:.4f}")

        print("\n--- Whisper Confidence Metrics ---")
        print(f"Average Log Probability: {result['avg_logprob']}")
        print(
            f"Average No Speech Probability: "
            f"{result['avg_no_speech_prob']}"
        )
        print(f"Number of Segments: {result['num_segments']}")
        print(f"Audio Duration: {result['duration']} seconds")

    try:
        # Monitoring mode: all non-empty transcripts still reach the
        # dialogue. Whisper's confidence now influences re-asking there.
        run_conversation(
            agent,
            lambda: listen(
                whisper_model,
                speaker_model,
                reference_embedding,
                rolling_verifier,
            ),
            report,
        )

    finally:
        # Always release audio, even after an unexpected error.
        try:
            sd.stop()
        except Exception:
            traceback.print_exc()
        try:
            rolling_verifier.end_turn()
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAssistant stopped.")
