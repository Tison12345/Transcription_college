
import re
import traceback
from enum import Enum, auto

from dialogue import asr_confidence
from dialogue.ack_validator import (
    fallback_acknowledgment,
    validate_acknowledgment,
)
from dialogue.call_state import (
    CallState,
    COMPLETED,
    ENDED_ERROR,
    ENDED_NO_RESPONSE,
)
from dialogue.llm import generate_acknowledgment
from output.tts import speak

# --- Retry / error limits (configurable) ------------------------------------

# Consecutive "no usable answer to this question" events (no speech,
# unintelligible, untrusted transcript) before the call is closed politely.
MAX_UNUSABLE_ATTEMPTS = 3

# Consecutive technical failures (capture / transcription / unexpected
# errors) before the call is closed.
MAX_SYSTEM_ERRORS = 2

# Consecutive text-to-speech failures before the call is closed.
MAX_TTS_FAILURES = 2

REASK_MESSAGES = [
    "Sorry, I didn't catch that.",
    "Let me try once more. Could you answer in a few simple words?",
]
NO_RESPONSE_CLOSING = (
    "I'm having trouble hearing you right now, so I'll end the call here. "
    "We will follow up another time. Goodbye!"
)
TECHNICAL_RETRY = "Sorry, I had a small technical problem."
TECHNICAL_CLOSING = (
    "I'm sorry, I'm having technical difficulties, so I'll end the call "
    "here. We will follow up another time. Goodbye!"
)

BARE_ANSWERS = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay",
    "no", "nope", "nah", "not really",
}

UNSURE_PATTERN = (r"\b(?:not sure|unsure|dont know|"
                  r"no idea|cant say|cant tell)\b")

class State(Enum):
    INTRODUCTION = auto()
    GENERAL_FEELING = auto()
    PAIN_ASSESSMENT = auto()
    PAIN_LOCATION = auto()
    CLOSING = auto()
    FINISHED = auto()


# --- Simple keyword rules (no medical interpretation, no diagnosis) ---------

PAIN_WORDS = (
    r"pain\w*|hurt\w*|ache\w*|aching|sore\w*|discomfort|uncomfortable|"
    r"cramp\w*|swell\w*|swollen|tender\w*|stiff\w*|throb\w*"
)

# Phrases (after normalize()) that mean the patient denies pain.
NO_PAIN_PATTERNS = [
    rf"\b(no|without|zero)\s+(?:\w+\s+)?(?:{PAIN_WORDS})\b",
    rf"\b(not|dont|doesnt|didnt|havent|never)\s+(?:\w+\s+){{0,2}}?"
    rf"(?:any|much)\s+(?:\w+\s+)?(?:{PAIN_WORDS})\b",
    r"\bnot in (?:any )?(?:pain|discomfort)\b",
    r"\b(?:nothing|nowhere)\s+(?:hurts?|aches?)\b",
    r"\b(?:not|dont|doesnt|didnt)\s+(?:hurt\w*|ache\w*)\b",
    r"\bpain ?free\b",
]

# Specific body locations. Broad words such as "foot" or "leg" are left out
# on purpose, so that a patient who says "pain in my foot" is asked where.
SPECIFIC_LOCATIONS = (
    r"ankles?|toes?|toenails?|calf|calves|heels?|knees?|shins?|thighs?|"
    r"hips?|back|arch|soles?|arms?|hands?|wrists?|shoulders?|neck|head|"
    r"chest|stomach|elbows?|fingers?|incision|wound|scar"
)

# Replies that are a clear "I'm fine / no problem" without mentioning pain.
FINE_PATTERN = (
    r"\b(fine|good|great|well|okay|ok|better|alright|nothing|none|nope|no|"
    r"nah|not really|nothing much)\b"
)


class DialogueStateMachine:

    def __init__(self):
        self.state = State.INTRODUCTION
        self.prompt_spoken = False

        # One-off neutral question that replaces the normal one (used when the
        # patient's answer was unclear). Cleared after the next answer.
        self.pending_prompt = None
        self.asked_unclear_followup = False

        # What the call has learned so far (in memory only).
        self.call = CallState()

        # Retry bookkeeping. Each counter covers one kind of problem and is
        # reset when that problem goes away.
        self.unusable_attempts = 0     # for the CURRENT question
        self.borderline_asked = False  # neutral re-ask used for this question
        self.system_errors = 0
        self.tts_failures = 0

        # Injectable for tests.
        self.speak_fn = speak

        self.questions = {
            State.INTRODUCTION:
                "Hello! I'm calling for your patient follow-up. "
                "Is this a good time to talk?",

            State.GENERAL_FEELING:
                "How have you been feeling recently?",

            State.PAIN_ASSESSMENT:
                "Are you experiencing any pain or discomfort?",

            State.PAIN_LOCATION:
                "Could you describe exactly where the pain or discomfort is?",

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
        }

        # Fixed transitions. GENERAL_FEELING and PAIN_ASSESSMENT are decided
        # from the patient's answer in choose_next_state().
        self.next_state = {
            State.INTRODUCTION: State.GENERAL_FEELING,
            State.PAIN_LOCATION: State.CLOSING,
            State.CLOSING: State.FINISHED,
        }

    def get_prompt(self):
        """Return the current question."""

        if self.state == State.FINISHED:
            return None

        return self.pending_prompt or self.questions[self.state]

    # --- Reading the patient's answer ---------------------------------------

    def mentions_pain(self, text):
        """True if the patient reports pain/discomfort (and not 'no pain')."""

        text = self.normalize(text)

        if any(re.search(p, text) for p in NO_PAIN_PATTERNS):
            return False

        return re.search(rf"\b(?:{PAIN_WORDS})\b", text) is not None

    def denies_pain(self, text):
        text = self.normalize(text)
        return any(re.search(p, text) for p in NO_PAIN_PATTERNS)

    @staticmethod
    def mentions_specific_location(text):
        return re.search(rf"\b(?:{SPECIFIC_LOCATIONS})\b",
                         DialogueStateMachine.normalize(text)) is not None

    def is_unclear_pain_answer(self, text):
        """Answer to the pain question that is neither pain nor 'no pain'."""

        return not (
            self.mentions_pain(text)
            or self.denies_pain(text)
            or re.search(FINE_PATTERN, self.normalize(text))
        )

    def choose_next_state(self, text):
        """Pick the next stage from what the patient actually said."""

        if self.state in (State.GENERAL_FEELING, State.PAIN_ASSESSMENT):
            if self.mentions_pain(text):
                if self.mentions_specific_location(text):
                    return State.CLOSING            # already described it
                return State.PAIN_LOCATION          # ask, never assume
            if self.state == State.GENERAL_FEELING:
                return State.PAIN_ASSESSMENT
            return State.CLOSING

        return self.next_state[self.state]

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

    # --- Speaking (TTS failures must not crash the call) --------------------

    def say(self, text):
        """
        Speak text. A TTS failure is logged and never raised; after
        MAX_TTS_FAILURES in a row the call is ended (we cannot talk).
        Returns True if the text was spoken.
        """

        if not text:
            return True

        try:
            self.speak_fn(text)
        except Exception:
            self.tts_failures += 1
            print(f"[error] text-to-speech failed "
                  f"({self.tts_failures}/{MAX_TTS_FAILURES}):")
            traceback.print_exc()

            if self.tts_failures >= MAX_TTS_FAILURES:
                self.call.conversation_status = ENDED_ERROR
                self.state = State.FINISHED
            return False

        self.tts_failures = 0
        return True

    # --- Retry limits and technical errors ----------------------------------

    def _reask(self, reason):
        """
        The patient gave no usable answer to the current question. Re-ask
        politely, or close the call after MAX_UNUSABLE_ATTEMPTS in a row.
        """

        self.unusable_attempts += 1
        print(f"[retry] no usable answer ({reason}): attempt "
              f"{self.unusable_attempts}/{MAX_UNUSABLE_ATTEMPTS}")

        if self.unusable_attempts >= MAX_UNUSABLE_ATTEMPTS:
            self.call.conversation_status = ENDED_NO_RESPONSE
            self.pending_prompt = NO_RESPONSE_CLOSING
            self.state = State.CLOSING
        else:
            index = min(self.unusable_attempts - 1, len(REASK_MESSAGES) - 1)
            self.pending_prompt = (
                f"{REASK_MESSAGES[index]} {self.questions[self.state]}"
            )

        self.prompt_spoken = False

    def handle_no_input(self):
        """No speech (or no usable transcript) was captured this turn."""

        if self.state in (State.FINISHED, State.CLOSING):
            return

        self._reask("no speech")

    def system_error(self, stage, error=None):
        """
        A technical failure (capture, transcription, unexpected error).
        The caller has already logged the exception. Apologise and retry
        the same question, or close the call after MAX_SYSTEM_ERRORS in a
        row. The patient never hears the technical details.
        """

        if self.state in (State.FINISHED, State.CLOSING):
            return

        self.system_errors += 1
        print(f"[error] {stage} failed "
              f"({self.system_errors}/{MAX_SYSTEM_ERRORS})"
              + (f": {type(error).__name__}: {error}" if error else ""))

        if self.system_errors >= MAX_SYSTEM_ERRORS:
            self.call.conversation_status = ENDED_ERROR
            self.pending_prompt = TECHNICAL_CLOSING
            self.state = State.CLOSING
        else:
            self.pending_prompt = (
                f"{TECHNICAL_RETRY} {self.questions[self.state]}"
            )

        self.prompt_spoken = False

    def turn_succeeded(self):
        """Call after a turn was handled without a technical failure."""
        self.system_errors = 0

    # --- CallState ----------------------------------------------------------

    @staticmethod
    def extract_location(text):
        match = re.search(
            rf"\b(?:{SPECIFIC_LOCATIONS})\b",
            DialogueStateMachine.normalize(text),
        )
        return match.group(0) if match else None

    def update_call_state(self, response, interpreted, uncertain=False):
        """
        Record what the patient said. Only called for transcripts we trust
        enough to act on. `uncertain` (a borderline transcript that was
        accepted) may fill empty fields but never overwrite known ones.
        """

        info = self.call.patient
        bare = self.normalize(response) in BARE_ANSWERS

        def put(field, value):
            if value in (None, ""):
                return
            if uncertain and getattr(info, field) not in (None, ""):
                return
            setattr(info, field, value)

        if self.state == State.GENERAL_FEELING and not bare:
            put("general_feeling", response)

        if self.state in (State.GENERAL_FEELING, State.PAIN_ASSESSMENT):
            if self.mentions_pain(interpreted):
                put("pain_reported", True)
                if not bare:
                    put("pain_description", response)
                put("pain_location", self.extract_location(response))
            elif (self.state == State.PAIN_ASSESSMENT
                  and self.denies_pain(interpreted)):
                put("pain_reported", False)

        elif self.state == State.PAIN_LOCATION:
            unsure = re.search(UNSURE_PATTERN, self.normalize(response))
            if not unsure and not bare:
                location = self.extract_location(response)
                # Keep the patient's words if no known body-part word.
                put("pain_location", location or response)
                put("pain_description", response)

    # --- Acknowledgments ----------------------------------------------------

    def acknowledge(self, interpreted):
        """
        Ask the LLM to word an acknowledgment, then validate it. A failed
        LLM call or an invalid reply is replaced by a safe fixed phrase. The
        LLM never chooses the next question.
        """

        if self.mentions_pain(interpreted):
            pain = True
        elif self.denies_pain(interpreted):
            pain = False
        else:
            pain = None

        try:
            candidate = generate_acknowledgment(
                interpreted, question=self.get_prompt()
            )
        except Exception:
            print("[error] LLM acknowledgment failed, using fallback:")
            traceback.print_exc()
            return fallback_acknowledgment(self.state.name, pain)

        valid, reason = validate_acknowledgment(candidate, interpreted, pain)

        if valid:
            return candidate

        print(f"[ack] rejected LLM reply ({reason}): {candidate!r}")
        return fallback_acknowledgment(self.state.name, pain)

    # --- Main entry ---------------------------------------------------------

    def process_response(self, response, confidence=None):
        """
        Process the patient's answer and control the conversation flow.

        confidence: optional dict with "avg_logprob" and
        "avg_no_speech_prob" from Whisper. If omitted, the transcript is
        trusted (old behaviour).
        """

        if self.state in (State.FINISHED, State.CLOSING):
            return

        if not response or not response.strip():
            print("No response detected.")
            self._reask("empty transcript")
            return

        response = response.strip()

        print(f"Patient response: {response}")

        # Handle requests to repeat the current question.
        if self.is_repeat_request(response):
            repeat_message = "Of course, let me repeat that."
            print("AI:", repeat_message)
            self.say(repeat_message)
            prompt = self.get_prompt()
            print("Assistant:", prompt)
            self.say(prompt)
            self.prompt_spoken = True
            return

        # How far to trust the transcript.
        confidence = confidence or {}
        verdict = asr_confidence.assess_transcript(
            confidence.get("avg_logprob"),
            confidence.get("avg_no_speech_prob"),
            short_valid_answer=self.normalize(response) in BARE_ANSWERS,
        )
        uncertain = False

        if verdict == asr_confidence.UNRELIABLE:
            self.call.low_confidence_turns += 1
            self.call.clarification_count += 1
            self._reask("unreliable transcript")
            return

        if verdict == asr_confidence.BORDERLINE:
            if not self.borderline_asked:
                # One neutral re-ask; the second attempt is accepted as-is.
                self.borderline_asked = True
                self.call.low_confidence_turns += 1
                self.call.clarification_count += 1
                self.pending_prompt = (
                    f"Sorry, could you say that once more? "
                    f"{self.questions[self.state]}"
                )
                self.prompt_spoken = False
                return
            uncertain = True

        interpreted = self.interpret_response(response)

        # Unclear answer to the pain question: one neutral rephrase, no
        # assumptions, then move on if it is still unclear.
        if (self.state == State.PAIN_ASSESSMENT
                and not self.asked_unclear_followup
                and self.is_unclear_pain_answer(interpreted)):
            self.asked_unclear_followup = True
            self.call.clarification_count += 1
            self.pending_prompt = (
                "Sorry, I didn't quite get that. Could you tell me a bit "
                "more about how you're feeling?"
            )
            self.prompt_spoken = False
            return

        # A usable answer: reset the retry counters for this question.
        self.unusable_attempts = 0
        self.borderline_asked = False

        self.update_call_state(response, interpreted, uncertain)

        ai_response = self.acknowledge(interpreted)

        print("AI:", ai_response)
        self.say(ai_response)

        if self.state == State.FINISHED:
            return                  # text-to-speech failed; call ended

        # Move to the next stage, based on what the patient said.
        self.state = self.choose_next_state(interpreted)
        if self.state == State.CLOSING:
            self.call.conversation_status = COMPLETED
        self.pending_prompt = None
        self.prompt_spoken = False


def main():

    agent = DialogueStateMachine()

    while agent.state != State.FINISHED:

        print(f"\nCurrent state: {agent.state.name}")
        if not agent.prompt_spoken:
            prompt = agent.get_prompt()
            print("Assistant:", prompt)
            agent.say(prompt)
            agent.prompt_spoken = True

        if agent.state == State.CLOSING:
            break

        response = input("Patient: ")

        agent.process_response(response)

    print("\nFollow-up conversation completed.")
    print(agent.call.to_dict())


if __name__ == "__main__":
    main()
