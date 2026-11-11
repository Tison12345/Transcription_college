"""
The conversation loop, separated from main.py so it can be tested with
fake listening and speaking (no microphone or models needed).

Every failure ends in one of three ways: the patient is asked again, the
call carries on, or the call is closed with a spoken goodbye. Exceptions
are always printed with their traceback, and the patient never hears them.
"""

import traceback

from dialogue.state_machine import State


def run_conversation(agent, listen_fn, report_fn=None):
    """
    Run the call until the state machine finishes.

    listen_fn() -> dict with "text", "avg_logprob", "avg_no_speech_prob"
                   (more keys are fine), or None if nothing usable was heard.
    report_fn(result) -> optional hook that prints diagnostics.
    """

    while agent.state != State.FINISHED:

        print(f"\nCurrent state: {agent.state.name}")

        if not agent.prompt_spoken:
            prompt = agent.get_prompt()
            print("Assistant:", prompt)
            agent.say(prompt)
            agent.prompt_spoken = True

            if agent.state == State.FINISHED:
                break               # text-to-speech keeps failing

        if agent.state == State.CLOSING:
            agent.state = State.FINISHED
            break

        try:
            result = listen_fn()

        except Exception as error:
            print("\n[error] listening failed:")
            traceback.print_exc()
            agent.system_error("listening", error)
            continue

        if result is None:
            print("No usable speech detected.")
            agent.handle_no_input()
            continue

        if report_fn is not None:
            try:
                report_fn(result)
            except Exception:
                print("[error] diagnostics printing failed (ignored):")
                traceback.print_exc()

        confidence = {
            "avg_logprob": result.get("avg_logprob"),
            "avg_no_speech_prob": result.get("avg_no_speech_prob"),
        }

        try:
            agent.process_response(result["text"], confidence)

        except Exception as error:
            print("\n[error] handling the response failed:")
            traceback.print_exc()
            agent.system_error("dialogue", error)
            continue

        agent.turn_succeeded()

    print("\nFollow-up conversation completed.")
    print(f"Call status: {agent.call.conversation_status}")
    return agent.call
