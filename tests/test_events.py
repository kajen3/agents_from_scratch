"""run_agent reports progress as typed events instead of printing."""

import subprocess
from collections.abc import Callable

import pytest

from agents_from_scratch import agent
from agents_from_scratch.agent import (
    AgentEvent,
    AssistantMessage,
    Final,
    Message,
    ToolEnd,
    ToolStart,
    TurnStart,
    run_agent,
)


def reply_with(*replies: str) -> Callable[[list[Message]], str]:
    it = iter(replies)
    return lambda _messages: next(it)


def test_event_sequence_for_one_bash_turn_then_exit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    llm = reply_with(
        "<action><bash>echo hi</bash><bash>exit 3</bash></action>",
        "<action><exit>done</exit></action>",
    )
    monkeypatch.setattr(agent, "llm_call", llm)
    events: list[AgentEvent] = []
    messages: list[Message] = [{"role": "user", "content": "go"}]

    answer, _ = run_agent(messages, max_turns=5, on_event=events.append)

    assert answer == "done"
    assert events == [
        TurnStart(0, 5),
        AssistantMessage("<action><bash>echo hi</bash><bash>exit 3</bash></action>"),
        ToolStart("echo hi"),
        ToolEnd("echo hi", 0, "hi\n"),
        ToolStart("exit 3"),
        ToolEnd("exit 3", 3, ""),
        TurnStart(1, 5),
        AssistantMessage("<action><exit>done</exit></action>"),
        Final("done"),
    ]
    assert capsys.readouterr().out == ""  # no printing without the CLI callback


def test_run_out_of_turns_still_emits_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent, "llm_call", reply_with("<bash>true</bash>"))
    events: list[AgentEvent] = []

    answer, _ = run_agent([], max_turns=1, on_event=events.append)

    assert events[-1] == Final(answer)


def test_timeout_still_closes_the_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise subprocess.TimeoutExpired("sleep", 120)

    monkeypatch.setattr(agent, "llm_call", reply_with("<bash>sleep 999</bash>"))
    monkeypatch.setattr(subprocess, "run", boom)
    events: list[AgentEvent] = []

    with pytest.raises(subprocess.TimeoutExpired):
        run_agent([], max_turns=1, on_event=events.append)

    assert isinstance(events[-2], ToolStart)
    assert isinstance(events[-1], ToolEnd) and events[-1].exit_code is None


def test_print_event_matches_cli_output(capsys: pytest.CaptureFixture[str]) -> None:
    agent.print_event(TurnStart(2, 9))
    agent.print_event(ToolStart("ls"))
    agent.print_event(Final("ignored"))

    assert capsys.readouterr().out == "Turn: 2/9\n$ ls\n"
