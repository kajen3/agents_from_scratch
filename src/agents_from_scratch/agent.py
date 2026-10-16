"""A minimal coding agent: an act -> observe -> decide loop over an LLM.

Each turn: ask the model for one or more actions, execute them (sequentially),
feed the results back. A turn is either one <exit> or one-or-more <bash>
actions; mixing the two is not allowed. The loop ends when the model
emits <exit> or the turn budget runs out.
"""

import argparse
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict, cast

import httpx2

from agents_from_scratch.banner import GREY, RESET, print_banner, prompt_text, use_color

# OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"
OLLAMA_ENDPOINT = "http://host.docker.internal:11434/api/chat"
MODEL = "qwen3.8:27b-mlx"

DEFAULT_MAX_TURNS = 25  # safety net; the agent is meant to stop itself with <exit>
MAX_OUTPUT_CHARS = 2000  # tool output is context we pay for on every later turn


class Message(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class TurnStart:
    turn: int
    max_turns: int


@dataclass(frozen=True)
class AssistantMessage:
    content: str


@dataclass(frozen=True)
class ToolStart:
    command: str


@dataclass(frozen=True)
class ToolEnd:
    command: str
    exit_code: int | None  # None: the command timed out or could not be run
    output: str  # full, untruncated; the model only sees MAX_OUTPUT_CHARS of it


@dataclass(frozen=True)
class Final:
    answer: str


AgentEvent = TurnStart | AssistantMessage | ToolStart | ToolEnd | Final
EventCallback = Callable[[AgentEvent], None]


SYSTEM_PROMPT = """You are an autonomous coding agent. You solve the user's task by taking one action at a time: you act, observe the result, then decide the next action.
Always explore your environment first. You're running on a computer and have access to a file-system as well as a bash-shell.

Every reply you send is a single <action> element. It contains either exactly one <exit> tag, or one or more <bash> tags. Do not mix <exit> with <bash> in the same action. Output only the <action> element, with nothing before or after it.

The two actions:
- <bash>: run a shell command. Put the command inside the tag. You may issue several <bash> tags in one action; each is executed in turn and every result is sent back to you as the next message, then you continue.
- <exit>: the task is done. This ends the loop. Put your final answer to the user inside the tag. An <exit> action contains no <bash> tags.

Examples:

# Run a command
<action>
<bash>ls -alh /path/to/dir</bash>
</action>

# Run another command
<action>
<bash>wc -l path/to/file.py</bash>
</action>

# Run several independent commands at once (each result comes back together)
<action>
<bash>ls -alh src</bash>
<bash>wc -l src/*.py</bash>
</action>

# Run a command with a pipe
<action>
<bash>grep -rn "ValueError" src | head -n 20</bash>
</action>

# Finish: report a fix
<action>
<exit>Fixed the ValueError in main.py by guarding against the empty list.</exit>
</action>

# Finish: answer a question
<action>
<exit>The project has 12 Python modules, the largest being agent.py at 240 lines.</exit>
</action>

# Finish: nothing to do
<action>
<exit>No changes were needed; the tests already pass.</exit>
</action>
"""


def llm_call(messages: list[Message]) -> str:
    r = httpx2.post(
        OLLAMA_ENDPOINT,
        json={"model": MODEL, "stream": False, "messages": messages},
        timeout=120,
    )
    r.raise_for_status()
    return cast(str, r.json()["message"]["content"])


def parse_xml(tag: str, string: str) -> list[str]:
    """All contents of <tag>...</tag>, in order of appearance, whitespace-stripped."""
    pattern = rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>"
    return [m.strip() for m in re.findall(pattern, string, re.DOTALL)]


def parse_action(text: str) -> dict[str, list[str]]:
    """Read the model's reply and return the action(s) it asked for.

    A turn is either a single <exit> or one-or-more <bash> actions. Every
    action in the reply must be of the same type; a reply that mixes <exit>
    with <bash> is invalid and is reported as an error. Returns
    {"exit": ["No action"]} when there is no recognised tag at all, so the
    loop always terminates cleanly.
    """
    exit_bodies = parse_xml("exit", text)
    bash_bodies = parse_xml("bash", text)
    if exit_bodies and bash_bodies:
        # Mixing the two action types is not allowed. Signal the mistake as
        # an error so the loop can feed it back and the model can retry,
        # rather than silently terminating.
        error = (
            "Invalid action: a turn may contain either a single <exit> tag "
            "or one or more <bash> tags, but not both. Re-issue the action "
            "with a single action type."
        )
        return {"error": [error]}
    if exit_bodies:
        return {"exit": exit_bodies}
    if bash_bodies:
        return {"bash": bash_bodies}
    return {"exit": ["No action"]}


def run_agent(
    messages: list[Message],
    max_turns: int,
    on_event: EventCallback | None = None,
) -> tuple[str, list[Message]]:
    """Run the loop; report progress through `on_event` instead of printing."""

    def emit(event: AgentEvent) -> None:
        if on_event is not None:
            on_event(event)

    for turn in range(max_turns):
        emit(TurnStart(turn, max_turns))
        response = llm_call(messages)
        emit(AssistantMessage(response))
        messages.append({"role": "assistant", "content": response})

        action = parse_action(response)
        if "exit" in action:
            emit(Final(action["exit"][0]))
            return action["exit"][0], messages
        if "error" in action:
            # The model mixed action types; tell it why and let it retry.
            messages.append({"role": "user", "content": action["error"][0]})
            continue
        if "bash" in action:
            # One or more commands, executed in turn; every result is fed
            # back together in a single observation message.
            observations: list[str] = []
            for command in action["bash"]:
                emit(ToolStart(command))
                try:
                    result = subprocess.run(
                        command,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=120,
                    )
                except Exception as e:
                    # Every ToolStart gets a ToolEnd, then fail as before.
                    emit(ToolEnd(command, None, str(e)))
                    raise
                emit(ToolEnd(command, result.returncode, result.stdout))
                observations.append(
                    f"Output of `$ {command}`:\n{result.stdout[:MAX_OUTPUT_CHARS]}\n(exit code {result.returncode})"
                )
            messages.append({"role": "user", "content": "\n\n".join(observations)})

    answer = f"Ran out of turns ({max_turns}) without an <exit> action."
    emit(Final(answer))
    return answer, messages


def print_event(event: AgentEvent) -> None:
    """Terminal rendering of agent events: turn and tool lines in dim grey."""
    match event:
        case TurnStart(turn, max_turns):
            line = f"Turn: {turn}/{max_turns}"
        case ToolStart(command):
            line = f"$ {command}"
        case _:
            return
    print(f"{GREY}{line}{RESET}" if use_color() else line)


def main(prompt: str | None, max_turns: int) -> None:
    stop: bool = False
    messages: list[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    if prompt:
        messages.append({"role": "user", "content": prompt})
        answer, _ = run_agent(messages, max_turns=max_turns, on_event=print_event)
        print("\n" + answer + "\n")
    else:
        print_banner(MODEL)  # interactive only; keeps -p output pipe-clean
        while not stop:
            prompt = input(prompt_text())
            if prompt == "exit":
                stop = True
            else:
                messages.append({"role": "user", "content": prompt})
                answer, messages = run_agent(
                    messages, max_turns=max_turns, on_event=print_event
                )
                print("\n" + answer + "\n")


def cli() -> None:
    parser = argparse.ArgumentParser("Coding agent from scratch")
    parser.add_argument("-p", "--prompt", required=False, type=str, help="the prompt")
    parser.add_argument(
        "-t",
        "--max-turns",
        default=DEFAULT_MAX_TURNS,
        type=int,
        help="max turns",
    )
    args = parser.parse_args()

    main(prompt=args.prompt, max_turns=args.max_turns)


if __name__ == "__main__":
    cli()
