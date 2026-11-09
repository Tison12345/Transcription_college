
import sounddevice as sd

from audio.utterance_detector import capture_one_utterance
from audio.transcribe import (
    load_whisper_model,
    transcribe_audio_with_confidence,
)
from audio.speaker_verification import (
    load_speaker_model,
    create_embedding,
    verify_speaker,
)
from dialogue.state_machine import DialogueStateMachine, State
from output.tts import speak


REFERENCE_AUDIO = "reference.wav"

NO_SPEECH_MESSAGE = (
    "Sorry, I didn't catch that. Could you please say that again?"
)


def listen(whisper_model, speaker_model, reference_embedding):
    """Capture speech, compare the voice, and transcribe it."""

    audio_file = capture_one_utterance()

    if audio_file is None:
        return None

    # Speaker verification
    print("\nChecking speaker similarity...")

    similarity = verify_speaker(
        speaker_model,
        reference_embedding,
        audio_file
    )

    print(f"Speaker similarity score: {similarity:.4f}")
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

    agent = DialogueStateMachine()

    print("\nPatient follow-up assistant is ready!")

    while agent.state != State.FINISHED:

        print(f"\nCurrent state: {agent.state.name}")

        if not agent.prompt_spoken:
            prompt = agent.get_prompt()
            print("Assistant:", prompt)
            speak(prompt)
            agent.prompt_spoken = True

        if agent.state == State.CLOSING:
            agent.state = State.FINISHED
            break

        try:
            result = listen(
                whisper_model,
                speaker_model,
                reference_embedding
            )

        except sd.PortAudioError as error:
            print(f"\nMicrophone error: {error}")
            print("Check the microphone and the INPUT_DEVICE setting.")
            return

        except Exception as error:
            print(f"\nProcessing error: {error}")
            return

        if result is None:
            print("No usable speech detected.")
            speak(NO_SPEECH_MESSAGE)
            continue

        transcript = result["text"]

        print(f"\nPatient: {transcript}")

        print("\n--- Speaker Verification ---")
        print(
            f"Similarity Score: "
            f"{result['speaker_similarity']:.4f}"
        )

        print("\n--- Whisper Confidence Metrics ---")
        print(f"Average Log Probability: {result['avg_logprob']}")
        print(
            f"Average No Speech Probability: "
            f"{result['avg_no_speech_prob']}"
        )
        print(f"Number of Segments: {result['num_segments']}")
        print(f"Audio Duration: {result['duration']} seconds")

        # Monitoring mode: all non-empty transcripts still reach the FSM.
        agent.process_response(transcript)

    print("\nFollow-up conversation completed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAssistant stopped.")
