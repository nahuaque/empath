# Empath

Empath is an experimental coaching chat app with a symbolic reasoning layer.
It combines an LLM with a miniKanren therapeutic reasoning kernel so responses
can be shaped by inspectable hypotheses, working-map memory, and small
coaching experiments.

Empath is not a diagnostic or clinical tool. It is intended for coaching,
reflection, emotional support, and structured problem solving.

## What It Does

- Runs a browser chat app with streaming responses.
- Tracks workspace-scoped context in a Working Map.
- Lets one user manage multiple workspaces and conversations.
- Explains why an intervention was chosen only when you click `Why this?`.
- Supports reflective listening, tiny experiments, and trace/debug inspection.
- Uses ACT, CBT, REBT, DBT, MBSR, Focusing, goal-direction, and consultative
  facilitation lenses as tentative reasoning systems.

For implementation details, see [docs/technical-reference.md](docs/technical-reference.md).

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- A DeepSeek API key for live model calls

## Setup

Clone the repo and install dependencies:

```bash
uv sync
```

Create a local `.env` file:

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
DEEPSEEK_API_KEY=your_key_here
```

Empath also still supports `DEEPSEEK_API_KEY` from the process environment and
the legacy `.deepseek_api_key` file.

## Run The Chat App

Start the API and browser chat app:

```bash
uv run empath-api
```

Open:

```text
http://127.0.0.1:8000
```

For a no-network local smoke test:

```bash
uv run empath-api --dry-run --store-backend memory
```

## Run The CLI

Interactive chat:

```bash
uv run empath
```

One-shot prompt:

```bash
uv run empath --once "I keep avoiding the investor update."
```

Dry-run the kernel-guided prompt path without calling DeepSeek:

```bash
uv run empath --dry-run --once "I'm sad today."
```

## Run The Kernel Demo

```bash
uv run empath-kernel-demo
```

## Storage

By default, the API persists local state through a SurrealDB-backed local file
at `.empath_surreal.db`. For throwaway sessions, use:

```bash
uv run empath-api --store-backend memory
```

For the legacy JSON snapshot backend:

```bash
uv run empath-api --store-backend json --state-file .empath_chat_state.json
```

## Development Checks

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Build the package:

```bash
uv build
```
