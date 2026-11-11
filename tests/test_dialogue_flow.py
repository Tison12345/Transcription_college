"""
Tests for the context-aware dialogue flow (dialogue/state_machine.py).

The LLM and TTS modules are replaced with fakes, so no model is loaded and
nothing is spoken. Run from the project root:

    python -m unittest tests.test_dialogue_flow -v
"""

import sys
import types
import unittest

# Fake heavy modules BEFORE importing the state machine.
fake_llm = types.ModuleType("dialogue.llm")
fake_llm.generate_acknowledgment = lambda text, question=None: "ACK"
fake_tts = types.ModuleType("output.tts")
spoken = []
fake_tts.speak = lambda text: spoken.append(text)
sys.modules["dialogue.llm"] = fake_llm
sys.modules["output.tts"] = fake_tts

from dialogue.state_machine import DialogueStateMachine, State  # noqa: E402
import dialogue.state_machine as _sm  # noqa: E402

# Route speech to this module's list even if another test module imported
# the state machine first with a different fake.
_sm.speak = lambda text: spoken.append(text)

BANNED = ("ankle", "toe", "calf", "exercise")


def agent_at(state):
    agent = DialogueStateMachine()
    agent.state = state
    return agent


def run(*answers, start=State.INTRODUCTION):
    """Feed answers; return (agent, [prompt asked before each answer])."""
    agent = agent_at(start)
    prompts = []
    for answer in answers:
        prompts.append(agent.get_prompt())
        agent.process_response(answer)
    return agent, prompts


class TestDialogueFlow(unittest.TestCase):

    def setUp(self):
        spoken.clear()

    def all_text(self, agent):
        return " ".join(list(agent.questions.values()) + spoken).lower()

    def test_exercise_states_are_gone(self):
        names = {s.name for s in State}
        self.assertFalse({"ANKLE_EXERCISE", "TOE_EXERCISE",
                          "CALF_EXERCISE"} & names)
        for q in DialogueStateMachine().questions.values():
            for word in BANNED:
                self.assertNotIn(word, q.lower())

    def test_1_feeling_fine_goes_to_pain_question_not_exercises(self):
        a = agent_at(State.GENERAL_FEELING)
        a.process_response("I am feeling fine.")
        self.assertEqual(a.state, State.PAIN_ASSESSMENT)

    def test_2_general_discomfort_asks_where(self):
        a = agent_at(State.GENERAL_FEELING)
        a.process_response("I have some discomfort.")
        self.assertEqual(a.state, State.PAIN_LOCATION)
        self.assertEqual(
            a.get_prompt(),
            "Could you describe exactly where the pain or discomfort is?")

    def test_3_foot_pain_without_location_asks_neutral_question(self):
        for state in (State.GENERAL_FEELING, State.PAIN_ASSESSMENT):
            a = agent_at(state)
            a.process_response("I have some pain in my foot.")
            self.assertEqual(a.state, State.PAIN_LOCATION)
            prompt = a.get_prompt().lower()
            for word in BANNED + ("foot", "toe"):
                self.assertNotIn(word, prompt)

    def test_4_explicit_ankle_pain_is_not_re_asked(self):
        a = agent_at(State.PAIN_ASSESSMENT)
        a.process_response("I have pain in my ankle.")
        self.assertEqual(a.state, State.CLOSING)

    def test_5_explicit_toe_pain_is_not_re_asked(self):
        a = agent_at(State.PAIN_ASSESSMENT)
        a.process_response("My toe hurts.")
        self.assertEqual(a.state, State.CLOSING)

    def test_6_no_pain_variants_go_to_closing(self):
        for text in ("No, I don't have any pain.", "No pain at all.",
                     "I am not in pain.", "Nothing hurts.",
                     "I'm not experiencing any pain or discomfort.",
                     "I feel fine, no discomfort."):
            a = agent_at(State.PAIN_ASSESSMENT)
            a.process_response(text)
            self.assertEqual(a.state, State.CLOSING, text)

    def test_7_short_yes_no(self):
        a = agent_at(State.PAIN_ASSESSMENT)
        a.process_response("No")
        self.assertEqual(a.state, State.CLOSING)

        a = agent_at(State.PAIN_ASSESSMENT)
        a.process_response("Yes")                  # pain, location unknown
        self.assertEqual(a.state, State.PAIN_LOCATION)

        a = agent_at(State.INTRODUCTION)
        a.process_response("Yes")
        self.assertEqual(a.state, State.GENERAL_FEELING)
        a.process_response("No")                   # not a pain statement
        self.assertEqual(a.state, State.PAIN_ASSESSMENT)

    def test_unclear_pain_answer_is_asked_once_neutrally(self):
        a = agent_at(State.PAIN_ASSESSMENT)
        a.process_response("Hmm, it is complicated.")
        self.assertEqual(a.state, State.PAIN_ASSESSMENT)
        self.assertIn("more about how you're feeling", a.get_prompt())
        a.process_response("Hmm, hard to say.")    # still unclear: move on
        self.assertEqual(a.state, State.CLOSING)

    def test_location_answer_then_closing(self):
        a, prompts = run("I have some pain in my foot.", "On the outer side.",
                         start=State.PAIN_ASSESSMENT)
        self.assertEqual(a.state, State.CLOSING)
        self.assertIn("exactly where", prompts[1])

    def test_negation_does_not_hide_real_pain(self):
        a = agent_at(State.PAIN_ASSESSMENT)
        a.process_response("I am not feeling fine, I have pain in my foot.")
        self.assertEqual(a.state, State.PAIN_LOCATION)

    def test_full_conversations_never_mention_unsupported_parts(self):
        scripts = [
            ["Yes", "I am feeling fine.", "No pain.", ],
            ["Yes", "Fine.", "I have some pain in my foot.",
             "Near the heel."],
            ["Yes", "A bit of discomfort.", "Mostly at night."],
        ]
        for script in scripts:
            spoken.clear()
            agent = DialogueStateMachine()
            asked = []
            for answer in script:
                if agent.state in (State.CLOSING, State.FINISHED):
                    break
                asked.append(agent.get_prompt())
                agent.process_response(answer)
            text = " ".join(asked).lower()
            for word in BANNED:
                self.assertNotIn(word, text, script)
            self.assertNotEqual(agent.state, State.FINISHED)

    def test_repeat_request_still_works(self):
        a = agent_at(State.PAIN_LOCATION)
        a.process_response("Could you repeat that?")
        self.assertEqual(a.state, State.PAIN_LOCATION)
        self.assertIn("exactly where", spoken[-1])


if __name__ == "__main__":
    unittest.main()
