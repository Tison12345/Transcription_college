"""
Tests for: Whisper-confidence-aware dialogue, LLM acknowledgment
validation, structured CallState, and retry limits / error handling.

LLM, TTS, microphone and Whisper are all replaced by fakes. These tests
check logic only; they say nothing about how real Whisper, Qwen or Coqui
behave.

    python -m unittest tests.test_dialogue_robustness -v
"""

import contextlib
import io
import sys
import types
import unittest
from unittest import mock

# Fake heavy modules BEFORE importing the state machine (same as
# test_dialogue_flow). Harmless if another test module already did it.
if "dialogue.llm" not in sys.modules:
    fake_llm = types.ModuleType("dialogue.llm")
    fake_llm.generate_acknowledgment = lambda text, question=None: "ACK"
    sys.modules["dialogue.llm"] = fake_llm
if "output.tts" not in sys.modules:
    fake_tts = types.ModuleType("output.tts")
    fake_tts.speak = lambda text: None
    sys.modules["output.tts"] = fake_tts

from dialogue import asr_confidence as conf                       # noqa: E402
from dialogue import call_state as cs                             # noqa: E402
from dialogue import state_machine as sm                          # noqa: E402
from dialogue.ack_validator import (                              # noqa: E402
    fallback_acknowledgment,
    validate_acknowledgment,
)
from dialogue.call_loop import run_conversation                   # noqa: E402
from dialogue.state_machine import DialogueStateMachine, State   # noqa: E402

GOOD = {"avg_logprob": -0.4, "avg_no_speech_prob": 0.05}
BORDERLINE = {"avg_logprob": -0.9, "avg_no_speech_prob": 0.1}
BAD = {"avg_logprob": -1.3, "avg_no_speech_prob": 0.1}
NOISE = {"avg_logprob": -0.5, "avg_no_speech_prob": 0.8}

LLM = "dialogue.state_machine.generate_acknowledgment"


def make_agent(state=State.INTRODUCTION):
    agent = DialogueStateMachine()
    agent.state = state
    agent.spoken = []
    agent.speak_fn = agent.spoken.append
    return agent


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def ask(agent, text, confidence=GOOD, ack="Thank you for letting me know."):
    with mock.patch(LLM, return_value=ack) as llm, quiet():
        agent.process_response(text, confidence)
    return llm


# ---------------------------------------------------------------- confidence

class TestAssessTranscript(unittest.TestCase):

    def test_good_metrics_are_ok(self):
        self.assertEqual(conf.assess_transcript(-0.4, 0.05), conf.OK)

    def test_very_low_logprob_is_unreliable(self):
        self.assertEqual(conf.assess_transcript(-1.21, 0.11), conf.UNRELIABLE)

    def test_high_no_speech_is_unreliable(self):
        self.assertEqual(conf.assess_transcript(-0.3, 0.7), conf.UNRELIABLE)

    def test_middle_values_are_borderline(self):
        self.assertEqual(conf.assess_transcript(-0.9, 0.1), conf.BORDERLINE)
        self.assertEqual(conf.assess_transcript(-0.3, 0.5), conf.BORDERLINE)

    def test_missing_metrics_are_not_held_against_the_transcript(self):
        self.assertEqual(conf.assess_transcript(None, None), conf.OK)

    def test_short_valid_answer_ignores_logprob_but_not_noise(self):
        self.assertEqual(
            conf.assess_transcript(-1.6, 0.1, short_valid_answer=True),
            conf.OK)
        self.assertEqual(
            conf.assess_transcript(-0.5, 0.8, short_valid_answer=True),
            conf.UNRELIABLE)


class TestConfidenceInDialogue(unittest.TestCase):

    def test_uncertain_transcript_triggers_clarification_not_ack(self):
        agent = make_agent(State.PAIN_ASSESSMENT)
        llm = ask(agent, "my food hurts", BAD)

        llm.assert_not_called()                       # no ack from bad text
        self.assertEqual(agent.spoken, [])            # nothing spoken yet
        self.assertEqual(agent.state, State.PAIN_ASSESSMENT)   # no progress
        self.assertFalse(agent.prompt_spoken)
        self.assertIn("didn't catch", agent.get_prompt())
        self.assertIn("pain or discomfort", agent.get_prompt())  # re-asks
        self.assertEqual(agent.call.low_confidence_turns, 1)

    def test_short_yes_no_is_not_rejected(self):
        for answer in ("Yes", "No", "Okay"):
            agent = make_agent(State.INTRODUCTION)
            ask(agent, answer, {"avg_logprob": -1.5,
                                "avg_no_speech_prob": 0.1})
            self.assertEqual(agent.state, State.GENERAL_FEELING, answer)
            self.assertEqual(agent.call.low_confidence_turns, 0)

    def test_noise_like_short_answer_is_still_rejected(self):
        agent = make_agent(State.INTRODUCTION)
        ask(agent, "Yes", NOISE)
        self.assertEqual(agent.state, State.INTRODUCTION)

    def test_borderline_is_asked_once_then_accepted(self):
        agent = make_agent(State.PAIN_ASSESSMENT)
        ask(agent, "my ankle hurts", BORDERLINE)
        self.assertEqual(agent.state, State.PAIN_ASSESSMENT)
        self.assertIn("once more", agent.get_prompt())

        ask(agent, "my ankle hurts", BORDERLINE)       # same quality again
        self.assertEqual(agent.state, State.CLOSING)    # accepted, no loop
        self.assertEqual(agent.call.low_confidence_turns, 1)

    def test_missing_confidence_keeps_old_behaviour(self):
        agent = make_agent(State.PAIN_ASSESSMENT)
        with mock.patch(LLM, return_value="I'm sorry to hear that."), quiet():
            agent.process_response("My ankle hurts")
        self.assertEqual(agent.state, State.CLOSING)

    def test_repeat_request_still_works_with_confidence(self):
        agent = make_agent(State.PAIN_LOCATION)
        ask(agent, "Could you repeat that?", BAD)
        self.assertEqual(agent.state, State.PAIN_LOCATION)
        self.assertIn("exactly where", agent.spoken[-1])


# ---------------------------------------------------------------- validator

class TestValidator(unittest.TestCase):

    def ok(self, ack, patient="my foot hurts", pain=True):
        return validate_acknowledgment(ack, patient, pain)[0]

    def test_good_acknowledgments_pass(self):
        self.assertTrue(self.ok("I'm sorry to hear that your foot hurts."))
        self.assertTrue(self.ok("Thank you for letting me know.",
                                "yes", None))
        self.assertTrue(self.ok("I'm glad to hear you're feeling fine.",
                                "I'm feeling fine", None))
        self.assertTrue(self.ok("I'm glad you're not having any pain.",
                                "No, I am not experiencing any pain.", False))

    def test_questions_are_rejected(self):
        self.assertFalse(self.ok("I'm sorry. Does it hurt a lot?"))
        self.assertFalse(self.ok("Where does it hurt, I'm sorry to hear"))

    def test_advice_is_rejected(self):
        self.assertFalse(self.ok("I'm sorry to hear that, you should rest."))
        self.assertFalse(self.ok("I'm sorry, please see a doctor."))
        self.assertFalse(self.ok("That sounds sore, try some ibuprofen."))

    def test_invented_body_part_or_symptom_is_rejected(self):
        self.assertFalse(self.ok("I'm sorry your ankle hurts."))
        self.assertFalse(self.ok("I'm sorry to hear about the swelling."))
        self.assertFalse(self.ok("I'm sorry you have a fever."))

    def test_invented_number_is_rejected(self):
        self.assertFalse(self.ok("I'm sorry it has hurt for 3 days."))

    def test_too_long_is_rejected(self):
        long_ack = "I'm sorry to hear that " + "really " * 30 + "today."
        self.assertFalse(self.ok(long_ack))

    def test_empty_and_role_labels_are_rejected(self):
        self.assertFalse(self.ok(""))
        self.assertFalse(self.ok("   "))
        self.assertFalse(self.ok("Reply: I'm sorry to hear that."))
        self.assertFalse(self.ok("I'm sorry.\nPatient: ouch"))

    def test_unrelated_content_is_rejected(self):
        self.assertFalse(self.ok("The weather is lovely today."))

    def test_tone_must_match_pain(self):
        self.assertFalse(self.ok("That's wonderful news.", pain=True))
        self.assertFalse(self.ok("I'm sorry to hear that.",
                                 "I have no pain", False))

    def test_fallbacks_depend_on_context_and_are_valid(self):
        self.assertEqual(fallback_acknowledgment("PAIN_ASSESSMENT", True),
                         "I'm sorry to hear that.")
        self.assertEqual(fallback_acknowledgment("INTRODUCTION"),
                         "Thank you for letting me know.")
        self.assertEqual(fallback_acknowledgment("PAIN_LOCATION"),
                         "Thank you for sharing that.")
        self.assertEqual(fallback_acknowledgment("GENERAL_FEELING"),
                         "I understand.")
        for state, pain in (("INTRODUCTION", None), ("PAIN_LOCATION", True),
                            ("GENERAL_FEELING", None),
                            ("PAIN_ASSESSMENT", False)):
            text = fallback_acknowledgment(state, pain)
            self.assertNotIn("?", text)


class TestAcknowledgmentInDialogue(unittest.TestCase):

    def test_valid_llm_reply_is_spoken(self):
        agent = make_agent(State.PAIN_ASSESSMENT)
        ask(agent, "My foot hurts", ack="I'm sorry to hear that your foot hurts.")
        self.assertEqual(agent.spoken,
                         ["I'm sorry to hear that your foot hurts."])

    def test_invalid_llm_replies_never_reach_tts(self):
        bad_replies = [
            "Does it hurt a lot?",
            "I'm sorry, you should rest and ice it.",
            "I'm sorry your ankle hurts.",
            "",
        ]
        for reply in bad_replies:
            agent = make_agent(State.PAIN_ASSESSMENT)
            ask(agent, "My foot hurts", ack=reply)
            self.assertEqual(agent.spoken, ["I'm sorry to hear that."], reply)

    def test_llm_exception_uses_fallback_and_call_continues(self):
        agent = make_agent(State.GENERAL_FEELING)
        with mock.patch(LLM, side_effect=RuntimeError("boom")), quiet():
            agent.process_response("I'm feeling fine", GOOD)
        self.assertEqual(agent.spoken, ["I understand."])
        self.assertEqual(agent.state, State.PAIN_ASSESSMENT)

    def test_llm_text_cannot_change_the_next_state(self):
        agent = make_agent(State.PAIN_ASSESSMENT)
        ask(agent, "No", ack="Next, let's talk about your toe exercise.")
        self.assertEqual(agent.state, State.CLOSING)      # policy decides


# ---------------------------------------------------------------- CallState

class TestCallState(unittest.TestCase):

    def test_feeling_is_recorded(self):
        agent = make_agent(State.GENERAL_FEELING)
        ask(agent, "I'm feeling fine.")
        info = agent.call.patient
        self.assertEqual(info.general_feeling, "I'm feeling fine.")
        self.assertIsNone(info.pain_reported)

    def test_pain_with_location(self):
        agent = make_agent(State.PAIN_ASSESSMENT)
        ask(agent, "My ankle hurts")
        info = agent.call.patient
        self.assertTrue(info.pain_reported)
        self.assertEqual(info.pain_location, "ankle")
        self.assertEqual(info.pain_description, "My ankle hurts")
        self.assertEqual(agent.call.conversation_status, cs.COMPLETED)

    def test_pain_without_location_then_clarification(self):
        agent = make_agent(State.PAIN_ASSESSMENT)
        ask(agent, "I'm experiencing pain, but I'm not sure where exactly.")
        info = agent.call.patient
        self.assertTrue(info.pain_reported)
        self.assertIsNone(info.pain_location)
        self.assertEqual(agent.state, State.PAIN_LOCATION)
        self.assertEqual(agent.call.conversation_status, cs.IN_PROGRESS)

        ask(agent, "I'm not sure.")               # no information given
        self.assertIsNone(agent.call.patient.pain_location)
        self.assertTrue(agent.call.patient.pain_reported)   # preserved

    def test_location_answer_is_recorded(self):
        agent = make_agent(State.PAIN_ASSESSMENT)
        ask(agent, "I have pain in my foot")
        ask(agent, "Near the heel")
        self.assertEqual(agent.call.patient.pain_location, "heel")

    def test_bare_yes_and_no(self):
        yes = make_agent(State.PAIN_ASSESSMENT)
        ask(yes, "Yes")
        self.assertTrue(yes.call.patient.pain_reported)
        self.assertIsNone(yes.call.patient.pain_description)

        no = make_agent(State.PAIN_ASSESSMENT)
        ask(no, "No")
        self.assertFalse(no.call.patient.pain_reported)

    def test_uncertain_transcript_does_not_touch_call_state(self):
        agent = make_agent(State.PAIN_ASSESSMENT)
        before = agent.call.to_dict()
        ask(agent, "my toe hurts", BAD)
        after = agent.call.to_dict()
        for key in ("patient",):
            self.assertEqual(before[key], after[key])

    def test_borderline_accepted_turn_cannot_overwrite_known_data(self):
        agent = make_agent(State.PAIN_ASSESSMENT)
        agent.call.patient.pain_location = "ankle"
        agent.update_call_state("my toe hurts", "my toe hurts", uncertain=True)
        self.assertEqual(agent.call.patient.pain_location, "ankle")

        agent.call.patient.pain_location = None
        agent.update_call_state("my toe hurts", "my toe hurts", uncertain=True)
        self.assertEqual(agent.call.patient.pain_location, "toe")  # fills

    def test_failed_turns_do_not_erase_earlier_information(self):
        agent = make_agent(State.GENERAL_FEELING)
        ask(agent, "I'm feeling fine")
        agent.handle_no_input()
        self.assertEqual(agent.call.patient.general_feeling,
                         "I'm feeling fine")

    def test_to_dict_separates_patient_info_from_call_status(self):
        data = cs.CallState().to_dict()
        self.assertEqual(
            set(data["patient"]),
            {"general_feeling", "pain_reported", "pain_location",
             "pain_description"})
        self.assertEqual(
            set(data) - {"patient"},
            {"conversation_status", "clarification_count",
             "low_confidence_turns"})


# ------------------------------------------------------------- retry limits

class TestRetryLimits(unittest.TestCase):

    def test_no_input_reasks_twice_then_closes(self):
        agent = make_agent(State.GENERAL_FEELING)
        question = agent.questions[State.GENERAL_FEELING]

        agent.handle_no_input()
        self.assertIn(question, agent.get_prompt())
        self.assertEqual(agent.state, State.GENERAL_FEELING)

        agent.handle_no_input()
        self.assertIn("once more", agent.get_prompt())
        self.assertEqual(agent.state, State.GENERAL_FEELING)

        agent.handle_no_input()
        self.assertEqual(agent.state, State.CLOSING)
        self.assertEqual(agent.get_prompt(), sm.NO_RESPONSE_CLOSING)
        self.assertEqual(agent.call.conversation_status, cs.ENDED_NO_RESPONSE)

    def test_counter_resets_after_a_usable_answer(self):
        agent = make_agent(State.INTRODUCTION)
        agent.handle_no_input()
        agent.handle_no_input()
        self.assertEqual(agent.unusable_attempts, 2)

        ask(agent, "Yes")                                 # usable
        self.assertEqual(agent.unusable_attempts, 0)
        agent.handle_no_input()
        self.assertEqual(agent.unusable_attempts, 1)
        self.assertEqual(agent.state, State.GENERAL_FEELING)   # not closed

    def test_repeated_unreliable_transcripts_close_the_call(self):
        agent = make_agent(State.PAIN_ASSESSMENT)
        for _ in range(sm.MAX_UNUSABLE_ATTEMPTS):
            ask(agent, "mumble mumble", BAD)
        self.assertEqual(agent.state, State.CLOSING)
        self.assertEqual(agent.call.conversation_status, cs.ENDED_NO_RESPONSE)

    def test_mixed_failures_share_the_per_question_counter_only(self):
        agent = make_agent(State.PAIN_ASSESSMENT)
        agent.handle_no_input()
        ask(agent, "mumble", BAD)
        self.assertEqual(agent.unusable_attempts, 2)
        self.assertEqual(agent.system_errors, 0)           # separate counter

    def test_unclear_pain_followup_is_asked_only_once(self):
        agent = make_agent(State.PAIN_ASSESSMENT)
        ask(agent, "Hmm, it is complicated.")
        ask(agent, "Hmm, hard to say.")
        self.assertEqual(agent.state, State.CLOSING)
        self.assertEqual(agent.call.clarification_count, 1)

    def test_system_errors_retry_then_close(self):
        agent = make_agent(State.PAIN_ASSESSMENT)
        with quiet():
            agent.system_error("listening", RuntimeError("mic"))
        self.assertEqual(agent.state, State.PAIN_ASSESSMENT)
        self.assertIn("technical problem", agent.get_prompt())
        self.assertNotIn("RuntimeError", agent.get_prompt())

        with quiet():
            agent.system_error("listening", RuntimeError("mic"))
        self.assertEqual(agent.state, State.CLOSING)
        self.assertEqual(agent.call.conversation_status, cs.ENDED_ERROR)

    def test_turn_succeeded_resets_system_errors(self):
        agent = make_agent(State.PAIN_ASSESSMENT)
        with quiet():
            agent.system_error("listening")
        agent.turn_succeeded()
        with quiet():
            agent.system_error("listening")
        self.assertEqual(agent.state, State.PAIN_ASSESSMENT)


# ------------------------------------------------------------ conversation loop

class TestConversationLoop(unittest.TestCase):

    def run_loop(self, agent, listen_fn, max_calls=50):
        calls = []

        def counted():
            calls.append(1)
            if len(calls) > max_calls:
                raise AssertionError("conversation did not terminate")
            return listen_fn(len(calls))

        with quiet():
            run_conversation(agent, counted)
        return len(calls)

    def test_silent_patient_ends_the_call(self):
        agent = make_agent()
        listens = self.run_loop(agent, lambda n: None)
        self.assertEqual(listens, sm.MAX_UNUSABLE_ATTEMPTS)
        self.assertEqual(agent.state, State.FINISHED)
        self.assertEqual(agent.call.conversation_status, cs.ENDED_NO_RESPONSE)
        self.assertEqual(agent.spoken[-1], sm.NO_RESPONSE_CLOSING)

    def test_persistent_listening_failure_ends_the_call_politely(self):
        agent = make_agent()

        def listen(n):
            raise RuntimeError("PortAudio exploded")

        listens = self.run_loop(agent, listen)
        self.assertEqual(listens, sm.MAX_SYSTEM_ERRORS)
        self.assertEqual(agent.call.conversation_status, cs.ENDED_ERROR)
        self.assertEqual(agent.spoken[-1], sm.TECHNICAL_CLOSING)
        for line in agent.spoken:                          # nothing technical
            self.assertNotIn("PortAudio", line)
            self.assertNotIn("RuntimeError", line)

    def test_one_transcription_failure_is_recovered(self):
        agent = make_agent()
        def listen(n):
            if n == 1:
                raise RuntimeError("asr failed once")
            return {"text": "Yes", **GOOD, "num_segments": 1, "duration": 1}

        with mock.patch(LLM, return_value="Thank you for letting me know."):
            self.run_loop(agent, listen, max_calls=6)
        self.assertTrue(any("technical problem" in s for s in agent.spoken))
        self.assertEqual(agent.call.conversation_status, cs.COMPLETED)
        self.assertEqual(agent.state, State.FINISHED)

    def test_dialogue_exception_is_handled_not_fatal(self):
        agent = make_agent(State.PAIN_ASSESSMENT)
        agent.choose_next_state = mock.Mock(side_effect=ValueError("bug"))

        def listen(n):
            return {"text": "No", **GOOD}

        listens = self.run_loop(agent, listen)
        self.assertEqual(listens, sm.MAX_SYSTEM_ERRORS)
        self.assertEqual(agent.call.conversation_status, cs.ENDED_ERROR)

    def test_tts_failure_ends_the_call_instead_of_looping(self):
        agent = make_agent()

        def broken(text):
            raise OSError("audio device missing")

        agent.speak_fn = broken

        def listen(n):
            return {"text": "Yes", **GOOD}

        with mock.patch(LLM, return_value="Thank you for letting me know."):
            self.run_loop(agent, listen, max_calls=10)
        self.assertEqual(agent.state, State.FINISHED)
        self.assertEqual(agent.call.conversation_status, cs.ENDED_ERROR)

    def test_single_tts_failure_is_survived(self):
        agent = make_agent()
        outcomes = iter([OSError("glitch")])

        def flaky(text):
            error = next(outcomes, None)
            if error:
                raise error
            agent.spoken.append(text)

        agent.speak_fn = flaky

        def listen(n):
            return {"text": "No", **GOOD}

        with mock.patch(LLM, return_value="Thank you for letting me know."):
            self.run_loop(agent, listen, max_calls=10)
        self.assertEqual(agent.call.conversation_status, cs.COMPLETED)

    def test_normal_call_still_completes(self):
        agent = make_agent()
        script = iter(["Yes", "I'm feeling fine", "My ankle hurts"])

        def listen(n):
            return {"text": next(script), **GOOD}

        with mock.patch(LLM, return_value="Thank you for letting me know."):
            self.run_loop(agent, listen)

        self.assertEqual(agent.state, State.FINISHED)
        self.assertEqual(agent.call.conversation_status, cs.COMPLETED)
        info = agent.call.patient
        self.assertEqual(info.general_feeling, "I'm feeling fine")
        self.assertTrue(info.pain_reported)
        self.assertEqual(info.pain_location, "ankle")
        asked = " ".join(agent.spoken).lower()
        for word in ("toe", "calf", "exercise"):
            self.assertNotIn(word, asked)


if __name__ == "__main__":
    unittest.main()
