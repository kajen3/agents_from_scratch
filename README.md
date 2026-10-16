<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/headline-dark.svg">
    <img src="docs/images/headline-light.svg" alt="koding aigent" height="48">
  </picture>
</h1>

A coding agent built from scratch: a small act → observe → decide loop that asks a
local LLM (served by Ollama) for a shell command, runs it, and feeds the output back
until the model declares the task done.

![koding aigent start-up banner](docs/images/startup-banner.png)

> [!CAUTION]
> **Never run this agent directly on your host system.**
> The agent executes model-written shell commands with `shell=True` and no
> filtering. A wrong or malicious command can delete files, read credentials, or
> exfiltrate data. Run it **only inside a container** (at the very least Docker;
> see [Run the agent](#run-the-agent)), on a throwaway clone of the repository,
> with nothing secret mounted or reachable.

## Status

This is a learning project that has grown into a *koding aigent that could, in
theory, do real work*. It is small enough to read in one sitting (~250 lines of
Python), but it is already used to develop itself: features such as parallel
(batched) tool calls were implemented by the agent working on its own repository.

What it does today:

- **XML action loop** – every reply is one `<action>` containing either `<bash>`
  (run a command, get the output back) or `<exit>` (final answer).
- **Interactive and one-shot modes** – a REPL with a coloured start-up banner, or
  `--prompt` for a single task whose answer is printed pipe-clean.
- **Bounded runs** – a turn budget (`--max-turns`), a 120 s timeout per command,
  and truncated tool output (2000 chars) so a runaway turn can't flood the context.
- **Parallel tool calls** – a turn may batch several independent `<bash>` actions;
  mixing `<bash>` and `<exit>` is rejected with an error fed back to the model.
  (Lives on the `multiple_tool_calls` branch; not yet merged into `main`.)
- **Hardened container** – read-only root filesystem, all capabilities dropped,
  non-root user, resource limits, and an independent clone as the only writable
  project state. Design: [`thoughts/shared/plans/001_hardened_agent_container_design.md`](thoughts/shared/plans/001_hardened_agent_container_design.md).
- **Quality gates** – ruff, pyright (strict), pytest, pre-commit, and CI.

Known limits: one model backend (Ollama), no file-edit tool beyond what bash
offers (arguably not a limit, see below), no persistence between sessions, and
the container still has network access to reach Ollama. Docker narrows the
blast radius; it is not a perfect sandbox. Review the agent's diff before you
commit or publish anything.

> **Is bash-only really a limitation?** Maybe not. In
> [*An Empirical Study of Harness Design for Coding Agents*](https://arxiv.org/abs/2609.20804)
> (Fan et al., 2026), models with good shell proficiency did as well or better with
> a bash-only action space than with predefined file/search/edit tools, at
> substantially lower cost, especially on command-line-centric tasks. Weaker models
> did worse without the structured tools, and results were mixed for some models
> (one dropped sharply on SWE-Bench but gained on Terminal-Bench), so the benefit
> depends on the model. The study did not evaluate the model used here.

The [`diary.md`](diary.md) logs pitfalls and design decisions made along the way.

## Develop

Development uses uv with Python 3.14.4; `uv sync` builds the environment. Ollama
on the host is reachable at `$OLLAMA_HOST` (`http://host.docker.internal:11434`),
so develop **inside a container** as well.

```bash
uv run ruff check .     # lint
uv run ruff format .    # format
uv run pyright          # type-check
uv run pytest           # test
```

## TUI

An interactive terminal UI built with [Textual](https://textual.textualize.io/):
conversation on the left, a live list of bash calls on the right (amber while
running, green on exit code 0, red on failure or timeout), a status line, and a
message box. Like the CLI, run it inside the container.

```sh
uv run agents-tui     # or: uv run python -m agents_from_scratch.tui
```

Enter sends, PgUp/PgDn scroll the conversation, Ctrl+C quits. The CLI
(`agents-from-scratch`, `--prompt`, `--max-turns`) is unchanged.

## TUI

An interactive terminal UI built with [Textual](https://textual.textualize.io/):
conversation on the left, a live list of bash calls on the right (amber while
running, green on exit code 0, red on failure or timeout), a status line, and a
message box. Like the CLI, run it inside the container.

```sh
uv run agents-tui     # or: uv run python -m agents_from_scratch.tui
```

Enter sends, PgUp/PgDn scroll the conversation, Ctrl+C quits. The CLI
(`agents-from-scratch`, `--prompt`, `--max-turns`) is unchanged.

## Run the agent

The agent runs in a hardened container. It works on an independent clone of
**any git repository** mounted at `/workspace`; the code driving the run is the
read-only copy of this harness baked into the image at `/app`. The agent never
touches your real working tree.

```sh
# 1. Build the image with the current harness (run from this repository).
docker build --tag agent_sandbox --file Dockerfile .

# 2. Pick the target repository and create a fresh, independent clone of it.
#    REPO can be a remote URL or a local path (local: only committed work is cloned).
REPO=https://github.com/<owner>/<repo>.git
WORKSPACE=/tmp/agent-workspace
rm -rf "$WORKSPACE"
git clone --no-hardlinks "$REPO" "$WORKSPACE"

# 3. Run one agent invocation against the clone.
docker run --rm \
    -it \
    --read-only \
    --tmpfs /tmp:size=64m \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --pids-limit 100 \
    --memory 4g \
    --cpus 2 \
    --add-host=host.docker.internal:host-gateway \
    --volume "$WORKSPACE":/workspace \
    --name agent-sandbox \
    agent_sandbox \
    /opt/app/bin/agents-from-scratch --max-turns 25 --prompt "..."

# 4. Review the agent's changes on the host.
git -C "$WORKSPACE" status
git -C "$WORKSPACE" diff
```

To work on this repository itself, use `REPO=.`.

Notes:

- Omit `--prompt` for the interactive session. Omit the trailing command to get an
  interactive shell in the same container instead.
- The image ships Python, `uv`, `curl`, `jq` and `vim`. If the target repository
  needs other toolchains, add them to the `Dockerfile`; the agent can't install
  system packages (non-root, read-only root filesystem).
- Only clone repositories you are willing to let the agent modify, and review the
  diff before pushing anything. Don't put credentials (SSH keys, tokens, `.env`
  files) into the clone.
- On a Linux host the clone must be writable by UID 1000, the container's `agent`
  user.
