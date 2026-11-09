
import re
from enum import Enum, auto
from dialogue.llm import generate_acknowledgment
from output.tts import speak

class State(Enum):
    INTRODUCTION = auto()
    GENERAL_FEELING = auto()
    PAIN_ASSESSMENT = auto()
    ANKLE_EXERCISE = auto()
    TOE_EXERCISE = auto()
    CALF_EXERCISE = auto()
    CLOSING = auto()
    FINISHED = auto()


class DialogueStateMachine:

    def __init__(self):
        self.state = State.INTRODUCTION
        self.prompt_spoken = False

        self.questions = {
            State.INTRODUCTION:
                "Hello! I'm calling for your patient follow-up. "
                "Is this a good time to talk?",

            State.GENERAL_FEELING:
                "How have you been feeling recently?",

            State.PAIN_ASSESSMENT:
                "Are you experiencing any pain or discomfort?",

            State.ANKLE_EXERCISE:
                "Let's discuss your prescribed ankle exercise. "
                "Have you been able to do it?",

            State.TOE_EXERCISE:
                "Have you been able to do your prescribed toe exercise?",

            State.CALF_EXERCISE:
                "Have you been able to do your prescribed calf exercise?",

            State.CLOSING:
                "Thank you for sharing your update. "
                "Please follow your clinician's instructions. Goodbye!",
        }

        # Meaning of a bare "yes"/"no" in each yes/no question state.
        self.answer_meanings = {
            State.INTRODUCTION: {
                "yes": "Yes, this is a good time to talk.",
                "no": "No, this is not a good time to talk.",
            },
            State.PAIN_ASSESSMENT: {
                "yes": "Yes, I am experiencing some pain or discomfort.",
                "no": "No, I am not experiencing any pain or discomfort.",
            },
            State.ANKLE_EXERCISE: {
                "yes": "Yes, I have been doing my ankle exercise.",
                "no": "No, I have not been able to do my ankle exercise.",
            },
            State.TOE_EXERCISE: {
                "yes": "Yes, I have been doing my toe exercise.",
                "no": "No, I have not been able to do my toe exercise.",
            },
            State.CALF_EXERCISE: {
                "yes": "Yes, I have been doing my calf exercise.",
                "no": "No, I have not been able to do my calf exercise.",
            },
        }

        self.next_state = {
            State.INTRODUCTION: State.GENERAL_FEELING,
            State.GENERAL_FEELING: State.PAIN_ASSESSMENT,
            State.PAIN_ASSESSMENT: State.ANKLE_EXERCISE,
            State.ANKLE_EXERCISE: State.TOE_EXERCISE,
            State.TOE_EXERCISE: State.CALF_EXERCISE,
            State.CALF_EXERCISE: State.CLOSING,
            State.CLOSING: State.FINISHED,
        }

    def get_prompt(self):
        """Return the current question."""

        if self.state == State.FINISHED:
            return None

        return self.questions[self.state]

    @staticmethod
    def normalize(text):
        """Lowercase, drop apostrophes and punctuation, collapse spaces."""

        text = text.lower().replace("'", "").replace("’", "")
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return text.strip()

    def is_repeat_request(self, response):
        """Check whether the patient wants the question repeated."""

        text = self.normalize(response)

        # Short utterances that only count when they are the whole reply.
        if text in {
            "what", "sorry", "pardon", "huh", "repeat", "again",
            "excuse me", "come again", "pardon me",
        }:
            return True

        # Phrases matched on word boundaries inside longer replies.
        repeat_phrases = [
            "repeat that", "repeat it", "repeat the question",
            "repeat please", "please repeat", "you repeat",
            "say that again", "say it again", "say again",
            "what was that", "what did you say", "come again", "pardon",
            "didnt hear", "couldnt hear", "did not hear", "cannot hear",
            "didnt understand", "did not understand", "dont understand",
            "didnt catch", "did not catch",
        ]

        padded = f" {text} "

        return any(f" {phrase} " in padded for phrase in repeat_phrases)

    def interpret_response(self, response):
        """Expand bare yes/no answers using the current question."""

        text = self.normalize(response)

        if text in {"no", "nope", "nah", "not really"}:
            return self.answer_meanings.get(self.state, {}).get("no", response)

        if text in {"yes", "yeah", "yep", "yup", "sure", "ok", "okay"}:
            return self.answer_meanings.get(self.state, {}).get("yes", response)

        return response

    def process_response(self, response):
        """Process patient response and control conversation flow."""

        if self.state == State.FINISHED:
            return

        if not response or not response.strip():
            print("No response detected. Please try again.")
            self.prompt_spoken = False
            return

        response = response.strip()

        print(f"Patient response: {response}")

        # Handle requests to repeat the current question.
        if self.is_repeat_request(response):
            repeat_message = "Of course, let me repeat that."
            print("AI:", repeat_message)
            speak(repeat_message)
            prompt = self.get_prompt()
            print("Assistant:", prompt)
            speak(prompt)
            self.prompt_spoken = True
            return

        # Generate a short empathetic acknowledgment.
        ai_response = generate_acknowledgment(
            self.interpret_response(response),
            question=self.get_prompt(),
        )

        print("AI:", ai_response)
        speak(ai_response)

        # Move to the next protocol state.
        self.state = self.next_state[self.state]
        self.prompt_spoken = False


def main():

    agent = DialogueStateMachine()

    while agent.state != State.FINISHED:

        print(f"\nCurrent state: {agent.state.name}")
        if not agent.prompt_spoken:
            prompt = agent.get_prompt()
            print("Assistant:", prompt)
            speak(prompt)
            agent.prompt_spoken = True

        response = input("Patient: ")

        agent.process_response(response)

    print("\nFollow-up conversation completed.")


if __name__ == "__main__":
    main()
