"""Headless TUI tests: a fake agent function stands in for run_agent."""

import asyncio
import threading
import time

import pytest
from textual.pilot import Pilot
from textual.widgets import Input, Markdown, Static

from agents_from_scratch import agent
from agents_from_scratch.agent import (
    EventCallback,
    Final,
    Message,
    ToolEnd,
    ToolStart,
    TurnStart,
)
from agents_from_scratch.tui import AgentApp, ToolLine


def fake_agent(
    messages: list[Message], max_turns: int, on_event: EventCallback | None = None
) -> tuple[str, list[Message]]:
    assert on_event is not None
    on_event(TurnStart(0, max_turns))
    on_event(ToolStart("echo hi"))
    on_event(ToolEnd("echo hi", 0, "hi\n"))
    on_event(ToolStart("false"))
    on_event(ToolEnd("false", 1, ""))
    on_event(Final("**all done**"))
    messages.append({"role": "assistant", "content": "all done"})
    return "**all done**", messages


def broken_agent(
    messages: list[Message], max_turns: int, on_event: EventCallback | None = None
) -> tuple[str, list[Message]]:
    raise RuntimeError("ollama unreachable")


async def wait_for_agent(app: AgentApp) -> None:
    thread = app.turn_thread
    if thread is not None:
        # call_from_thread needs the loop, so join off-loop
        await asyncio.to_thread(thread.join, 5)
        assert not thread.is_alive()


async def submit(app: AgentApp, pilot: Pilot[None], text: str) -> None:
    app.query_one(Input).value = text
    await pilot.press("enter")
    await wait_for_agent(app)
    await pilot.pause()


async def test_prompt_runs_agent_and_shows_results() -> None:
    app = AgentApp(agent_fn=fake_agent)
    async with app.run_test() as pilot:
        await submit(app, pilot, "hello")

        texts = [str(w.render()) for w in app.query("#conversation Static")]
        assert "hello" in texts
        assert (
            app.query_one("#conversation Markdown", Markdown).source == "**all done**"
        )
        tools = list(app.query(ToolLine))
        assert [t.command for t in tools] == ["echo hi", "false"]
        assert tools[0].has_class("-ok") and not tools[0].has_class("-running")
        assert tools[1].has_class("-err")
        assert str(app.query_one("#status", Static).render()) == "Idle"
        # history persists: system prompt, user message, agent's addition
        assert [m["role"] for m in app.messages] == ["system", "user", "assistant"]
        assert app.query_one(Input).value == ""


async def test_empty_input_is_ignored() -> None:
    app = AgentApp(agent_fn=fake_agent)
    async with app.run_test() as pilot:
        await submit(app, pilot, "   ")
        assert len(app.messages) == 1
        assert not app.query(ToolLine)


async def test_tool_is_running_until_it_ends_and_input_is_ignored_meanwhile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()
    started = threading.Event()
    released_in_time: list[bool] = []

    def slow_agent(
        messages: list[Message], max_turns: int, on_event: EventCallback | None = None
    ) -> tuple[str, list[Message]]:
        assert on_event is not None
        on_event(ToolStart("sleep 1"))  # returns once the UI has handled it
        started.set()
        released_in_time.append(release.wait(5))
        on_event(ToolEnd("sleep 1", 0, ""))
        return "ok", messages

    app = AgentApp(agent_fn=slow_agent)
    async with app.run_test() as pilot:
        app.query_one(Input).value = "go"
        await pilot.press("enter")
        assert await asyncio.to_thread(started.wait, 5)
        await pilot.pause()
        (tool,) = app.query(ToolLine)
        assert tool.has_class("-running")
        assert str(app.query_one("#status", Static).render()).startswith("Running")

        notified: list[str] = []

        def fake_notify(message: str, **_kwargs: object) -> None:
            notified.append(message)

        monkeypatch.setattr(app, "notify", fake_notify)
        app.query_one(Input).value = "second"
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one(Input).value == "second"  # ignored, text kept
        assert len(app.messages) == 2
        assert notified == ["Agent is busy…"]

        release.set()
        await wait_for_agent(app)
        await pilot.pause()
        assert tool.has_class("-ok")
    assert released_in_time == [True]


async def test_agent_error_is_shown_and_app_returns_to_idle() -> None:
    app = AgentApp(agent_fn=broken_agent)
    async with app.run_test() as pilot:
        await submit(app, pilot, "hello")

        errors = [str(w.render()) for w in app.query(".error")]
        assert any("ollama unreachable" in e for e in errors)
        assert str(app.query_one("#status", Static).render()) == "Idle"
        assert not app.running


async def test_ctrl_c_exits() -> None:
    app = AgentApp(agent_fn=fake_agent)
    async with app.run_test() as pilot:
        await pilot.press("ctrl+c")
        await pilot.pause()
    assert not app.is_running


async def test_pageup_scrolls_conversation_while_input_is_focused() -> None:
    app = AgentApp(agent_fn=fake_agent)
    async with app.run_test(size=(80, 12)) as pilot:
        for i in range(10):
            await app.add_to_conversation(Static(f"line {i}\n\n\n"))
        await pilot.pause()
        conversation = app.query_one("#conversation")
        assert conversation.scroll_y > 0  # auto-scrolled to the newest content
        before = conversation.scroll_y
        await pilot.press("pageup")
        await pilot.pause()
        assert conversation.scroll_y < before


async def test_ctrl_c_mid_run_exits_promptly() -> None:
    never = threading.Event()  # released only at teardown
    started = threading.Event()

    def blocked_agent(
        messages: list[Message], max_turns: int, on_event: EventCallback | None = None
    ) -> tuple[str, list[Message]]:
        started.set()
        never.wait(30)
        return "late", messages

    app = AgentApp(agent_fn=blocked_agent)
    try:
        t0 = time.monotonic()
        async with app.run_test() as pilot:
            app.query_one(Input).value = "go"
            await pilot.press("enter")
            assert await asyncio.to_thread(started.wait, 5)
            await pilot.press("ctrl+c")
            await pilot.pause()
        assert time.monotonic() - t0 < 2
        assert not app.is_running
    finally:
        never.set()
        await wait_for_agent(app)  # joins off-loop so call_from_thread can finish


async def test_tall_markdown_answer_ends_scrolled_to_the_bottom() -> None:
    answer = "\n\n".join(f"paragraph {i}" for i in range(200))

    def tall_agent(
        messages: list[Message], max_turns: int, on_event: EventCallback | None = None
    ) -> tuple[str, list[Message]]:
        assert on_event is not None
        on_event(Final(answer))
        return answer, messages

    app = AgentApp(agent_fn=tall_agent)
    async with app.run_test(size=(80, 20)) as pilot:
        await submit(app, pilot, "go")
        conversation = app.query_one("#conversation")
        for _ in range(5):
            await pilot.pause(0.05)
            await pilot.wait_for_scheduled_animations()
        assert conversation.max_scroll_y > 0
        assert conversation.scroll_y == conversation.max_scroll_y


async def test_failed_turn_rolls_back_history() -> None:
    calls: list[list[Message]] = []

    def flaky_agent(
        messages: list[Message], max_turns: int, on_event: EventCallback | None = None
    ) -> tuple[str, list[Message]]:
        calls.append([dict(m) for m in messages])  # pyright: ignore[reportArgumentType]
        if len(calls) == 1:
            messages.append({"role": "assistant", "content": "partial"})
            raise RuntimeError("boom")
        return "ok", messages

    app = AgentApp(agent_fn=flaky_agent)
    async with app.run_test() as pilot:
        await submit(app, pilot, "first")
        assert [m["role"] for m in app.messages] == ["system"]
        await submit(app, pilot, "second")

    assert [(m["role"], m["content"]) for m in calls[1][1:]] == [("user", "second")]
    assert calls[1][0]["role"] == "system"
    assert len(calls[1]) == 2


async def test_history_persists_across_two_prompts() -> None:
    calls: list[list[Message]] = []

    def agent_fn(
        messages: list[Message], max_turns: int, on_event: EventCallback | None = None
    ) -> tuple[str, list[Message]]:
        calls.append([dict(m) for m in messages])  # pyright: ignore[reportArgumentType]
        messages.append({"role": "assistant", "content": f"reply{len(calls)}"})
        return "ok", messages

    app = AgentApp(agent_fn=agent_fn)
    async with app.run_test() as pilot:
        await submit(app, pilot, "one")
        await submit(app, pilot, "two")

    assert [(m["role"], m["content"]) for m in calls[1][1:]] == [
        ("user", "one"),
        ("assistant", "reply1"),
        ("user", "two"),
    ]
    assert calls[1][0]["role"] == "system"


async def test_error_flips_running_tool_line_to_err() -> None:
    def dying_agent(
        messages: list[Message], max_turns: int, on_event: EventCallback | None = None
    ) -> tuple[str, list[Message]]:
        assert on_event is not None
        on_event(ToolStart("sleep 999"))
        raise TimeoutError("timed out")

    app = AgentApp(agent_fn=dying_agent)
    async with app.run_test() as pilot:
        await submit(app, pilot, "go")
        (tool,) = app.query(ToolLine)
        assert tool.has_class("-err") and not tool.has_class("-running")


async def test_markup_in_user_text_commands_errors_and_answers_is_literal() -> None:
    nasty = "[red]x[/] and a stray [/] end"

    def nasty_agent(
        messages: list[Message], max_turns: int, on_event: EventCallback | None = None
    ) -> tuple[str, list[Message]]:
        assert on_event is not None
        on_event(ToolStart(f"echo {nasty}"))
        on_event(ToolEnd(f"echo {nasty}", 0, ""))
        on_event(Final(nasty))
        raise RuntimeError(nasty)

    app = AgentApp(agent_fn=nasty_agent)
    async with app.run_test() as pilot:
        await submit(app, pilot, nasty)

        texts = [str(w.render()) for w in app.query("#conversation Static")]
        assert nasty in texts
        assert any(nasty in t for t in texts if t.startswith("Error:"))
        (tool,) = app.query(ToolLine)
        assert nasty in str(tool.render())
        assert app.query_one("#conversation Markdown", Markdown).source == nasty


async def test_enter_sends_even_when_input_is_not_focused() -> None:
    app = AgentApp(agent_fn=fake_agent)
    async with app.run_test() as pilot:
        app.query_one("#conversation").focus()
        app.query_one(Input).value = "hello"
        await pilot.press("enter")
        await wait_for_agent(app)
        await pilot.pause()
        assert [m["role"] for m in app.messages] == ["system", "user", "assistant"]


def test_cli_main_prints_turns_tool_calls_and_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    replies = iter(
        [
            "<action><bash>echo hi</bash></action>",
            "<action><exit>all finished</exit></action>",
        ]
    )

    def fake_llm(_messages: list[Message]) -> str:
        return next(replies)

    monkeypatch.setattr(agent, "llm_call", fake_llm)
    monkeypatch.setenv("NO_COLOR", "1")

    agent.main("x", 3)

    out = capsys.readouterr().out
    assert "Turn: 0/3" in out
    assert "$ echo hi" in out
    assert "all finished" in out
    assert "\x1b[" not in out
