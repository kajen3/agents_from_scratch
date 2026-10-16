"""Headless TUI tests: a fake agent function stands in for run_agent."""

import threading

from textual.pilot import Pilot
from textual.widgets import Input, Markdown, Static

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
    await app.workers.wait_for_complete()  # pyright: ignore[reportUnknownMemberType]


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


async def test_tool_is_running_until_it_ends_and_input_is_ignored_meanwhile() -> None:
    release = threading.Event()

    def slow_agent(
        messages: list[Message], max_turns: int, on_event: EventCallback | None = None
    ) -> tuple[str, list[Message]]:
        assert on_event is not None
        on_event(ToolStart("sleep 1"))
        release.wait(5)
        on_event(ToolEnd("sleep 1", 0, ""))
        return "ok", messages

    app = AgentApp(agent_fn=slow_agent)
    async with app.run_test() as pilot:
        app.query_one(Input).value = "go"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        (tool,) = app.query(ToolLine)
        assert tool.has_class("-running")
        assert str(app.query_one("#status", Static).render()).startswith("Running")

        app.query_one(Input).value = "second"
        await pilot.press("enter")
        assert app.query_one(Input).value == "second"  # ignored, text kept
        assert len(app.messages) == 2

        release.set()
        await wait_for_agent(app)
        await pilot.pause()
        assert tool.has_class("-ok")


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
