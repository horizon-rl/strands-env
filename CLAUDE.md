# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Strands-env is an RL environment abstraction for Strands agents — step, observe, reward. It provides a base `Environment` class that wraps a Strands `Agent` with token-level observation tracking (TITO), reward computation, and termination handling. Supports SGLang, Bedrock, and OpenAI model backends.

## Commands

### Setup
```bash
pip install -e ".[dev]"
```

### Linting
```bash
ruff check src/
ruff format --check src/
```

### Testing
```bash
# Unit tests (no server needed)
pytest tests/unit/ -v

# Single test
pytest tests/unit/test_environment.py::TestStep::test_successful_step -v

# Unit tests with coverage
pytest tests/unit/ -v --cov=src/strands_env --cov-report=html

# Integration tests (requires running SGLang server; model ID auto-detected via /get_model_info)
# Tests skip automatically if server is unreachable (/health check)
pytest tests/integration/ -v --sglang-base-url=http://localhost:30000
# Or via env var: SGLANG_BASE_URL=http://localhost:30000 pytest tests/integration/
```

### Integration Tests with Remote GPU Server

```bash
# 1. Launch SGLang on the remote server in docker
ssh <remote-host> "sudo docker run -d --gpus '\"device=0\"' --name sglang-test -p 30000:30000 --ipc=host lmsysorg/sglang:<tag> python3 -m sglang.launch_server --model-path <model-id> --host 0.0.0.0 --port 30000 --tp <num_gpus> --mem-fraction-static 0.7"
# 2. Tunnel the port locally
ssh -L 30000:localhost:30000 -N -f <remote-host>
# 3. Run tests locally
pytest tests/integration/ -v
```

## Architecture

The package lives in `src/strands_env/core/` with three modules:

**types.py** — All data types. `Action` carries a user message + `TaskContext` (ground truth, conversation history, arbitrary metadata via `extra="allow"`). `Observation` holds messages, metrics, and optional `TokenObservation` for TITO training. `TerminationReason` maps agent exceptions to enum values via `from_error()` which walks exception cause chains. `StepResult` bundles observation + reward + termination reason.

**models.py** — `ModelFactory = Callable[[], Model]` type and three factory functions (`sglang_model_factory`, `bedrock_model_factory`, `openai_model_factory`). Each returns a zero-arg lambda that creates a fresh Model instance per `step()` call for concurrent isolation. Bedrock and OpenAI remap `max_new_tokens` → `max_tokens` with a shallow dict copy to avoid mutating defaults.

**environment.py** — Base `Environment` class. `step(action)` creates a fresh model via factory, attaches a `TokenManager`, builds an `Agent` with tools/hooks (always includes `ToolIterationLimiter`), runs `invoke_async`, then collects metrics and optional reward. Subclasses override `get_tools()` and `get_hooks()` to customize. Messages are sliced so only new messages from the current step appear in the observation.

### Key Design Decisions

- **Factory pattern**: `ModelFactory` returns lambdas (not Model instances) so each `step()` gets a fresh model with clean token tracking state.
- **TITO token tracking**: `TokenManager` on SGLang models captures exact token IDs and logprobs during generation. `TokenObservation.from_token_manager()` extracts prompt/rollout split. Non-SGLang models get an empty `TokenManager` (returns `None` from `from_token_manager`).
- **`list()` copies**: Tools, hooks, and messages are copied via `list()` before passing to Agent to prevent cross-step mutation.
- **ToolIterationLimiter**: Always prepended to hooks list. Raises `MaxToolIterationsReachedError` which `TerminationReason.from_error()` maps to `MAX_TOOL_ITERATIONS_REACHED`.

## Code Style

- Ruff for linting and formatting (line-length 120, rules: E, F, I, N, W)
- Conventional commits (feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert)
- Python 3.10+ required
- asyncio_mode = "auto" for pytest-asyncio
- Async-first: all Environment methods that interact with Agent are async
