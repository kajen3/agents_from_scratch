"""Textual TUI for the agent: conversation, live tool calls, status, input.

`run_agent` is synchronous, so it runs in a background thread. Its events are
marshalled onto the UI thread with `call_from_thread`; widgets are only ever
touched from the UI thread.
"""

import threading
from collections.abc import Callable
from concurrent.futures import CancelledError
from typing import Protocol

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Footer, Input, Markdown, Static

from agents_from_scratch import agent
from agents_from_scratch.agent import (
    DEFAULT_MAX_TURNS,
    SYSTEM_PROMPT,
    AgentEvent,
    EventCallback,
    Final,
    Message,
    ToolEnd,
    ToolStart,
    TurnStart,
)

DOT_COLORS = {"-running": "yellow", "-ok": "green", "-err": "red"}


class AgentFn(Protocol):
    def __call__(
        self,
        messages: list[Message],
        max_turns: int,
        on_event: EventCallback | None = None,
    ) -> tuple[str, list[Message]]: ...


class ToolLine(Static):
    """One bash call: a coloured dot and the command on a single line.

    The state is a CSS class (`-running`, `-ok` or `-err`), so it is both
    styleable and assertable in tests.
    """

    def __init__(self, command: str) -> None:
        super().__init__(classes="-running")
        self.command = command
        self.set_state("-running")

    def set_state(self, state: str) -> None:
        self.remove_class(*DOT_COLORS)
        self.add_class(state)
        self.update(
            Text.assemble(
                ("● ", DOT_COLORS[state]),
                ("Bash ", "bold"),
                self.command.replace("\n", " "),
                no_wrap=True,
                overflow="ellipsis",
            )
        )


class AgentApp(App[None]):
    CSS = """
    #main { height: 1fr; }
    #conversation { width: 3fr; border: round $primary; padding: 0 1; }
    #tools { width: 1fr; border: round $primary; padding: 0 1; }
    .user { background: $boost; margin: 1 0; padding: 1 1; }
    .error { color: $error; margin: 1 0; }
    ToolLine { height: 1; }
    #status { height: 1; padding: 0 1; color: $text-muted; }
    #prompt { border: round $primary; }
    """

    BINDINGS = [
        Binding("enter", "submit", "Send"),
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("pageup", "scroll_conversation('up')", "PgUp/PgDn scroll"),
        Binding("pagedown", "scroll_conversation('down')", show=False),
    ]

    def __init__(
        self,
        agent_fn: AgentFn = agent.run_agent,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> None:
        super().__init__()
        self.agent_fn = agent_fn
        self.max_turns = max_turns
        self.messages: list[Message] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.running = False
        self.turn_thread: threading.Thread | None = None
        self._current_tool: ToolLine | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="main"):
            yield VerticalScroll(id="conversation")
            yield VerticalScroll(id="tools")
        yield Static("Idle", id="status")
        yield Input(placeholder="Send a message", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#conversation").border_title = "Conversation"
        self.query_one("#tools").border_title = "Tools"
        self.query_one("#prompt").border_title = "Message"
        self.query_one("#prompt").focus()
        # Stay scrolled to the bottom as content (incl. Markdown, which lays
        # out its children asynchronously) is added, until the user scrolls up.
        self.query_one("#conversation", VerticalScroll).anchor()

    # --- actions -----------------------------------------------------------

    async def action_submit(self) -> None:
        # The focused Input handles Enter itself; this makes Enter send the
        # typed message even when focus is elsewhere (e.g. the conversation).
        await self.query_one("#prompt", Input).action_submit()

    def action_scroll_conversation(self, direction: str) -> None:
        conversation = self.query_one("#conversation", VerticalScroll)
        if direction == "up":
            conversation.scroll_page_up()
        else:
            conversation.scroll_page_down()

    # --- user input --------------------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        if self.running:
            self.notify("Agent is busy…")  # keep the typed text
            return
        event.input.clear()
        await self.add_to_conversation(Static(text, classes="user", markup=False))
        history_length = len(self.messages)  # restore point if the turn fails
        self.messages.append({"role": "user", "content": text})
        self.set_status("Running…")
        self.running = True
        self.run_turns(history_length)

    # --- agent thread --------------------------------------------------------

    def run_turns(self, history_length: int) -> None:
        # A daemon thread rather than a Textual thread worker: workers are
        # joined on shutdown, so quitting mid-run would block until a slow
        # LLM call or command finished (up to 120 s). A thread can't be
        # cancelled, but a daemon one doesn't keep the process alive.
        def on_event(event: AgentEvent) -> None:
            self.call_from_thread(self.handle_event, event)

        def finish(callback: Callable[..., object], *args: object) -> None:
            try:
                self.call_from_thread(callback, *args)
            except RuntimeError, CancelledError, NoMatches:
                pass  # the app is already gone

        def run() -> None:
            try:
                self.agent_fn(self.messages, self.max_turns, on_event=on_event)
            except Exception as e:  # keep the app alive: show it and go idle
                finish(self.show_error, e, history_length)
            finally:
                finish(self.set_idle)

        self.turn_thread = threading.Thread(target=run, daemon=True)
        self.turn_thread.start()

    # --- UI-thread handlers ------------------------------------------------

    async def handle_event(self, event: AgentEvent) -> None:
        match event:
            case TurnStart(turn, max_turns):
                self.set_status(f"Running… turn {turn + 1}/{max_turns}")
            case ToolStart(command):
                self._current_tool = ToolLine(command)
                tools = self.query_one("#tools", VerticalScroll)
                await tools.mount(self._current_tool)
                tools.scroll_end(animate=False)
            case ToolEnd(_, exit_code):
                if self._current_tool is not None:
                    self._current_tool.set_state("-ok" if exit_code == 0 else "-err")
                    self._current_tool = None
            case Final(answer):
                await self.add_to_conversation(Markdown(answer, open_links=False))
            case _:
                pass

    async def show_error(self, error: Exception, history_length: int) -> None:
        del self.messages[history_length:]  # drop the failed turn's messages
        if self._current_tool is not None:  # the tool died mid-run
            self._current_tool.set_state("-err")
            self._current_tool = None
        message = f"Error: {type(error).__name__}: {error}"
        await self.add_to_conversation(Static(message, classes="error", markup=False))

    def set_idle(self) -> None:
        self.running = False
        self.set_status("Idle")

    def set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    async def add_to_conversation(self, widget: Static | Markdown) -> None:
        conversation = self.query_one("#conversation", VerticalScroll)
        await conversation.mount(widget)


def main() -> None:
    AgentApp().run()


if __name__ == "__main__":
    main()
