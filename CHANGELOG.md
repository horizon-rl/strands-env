# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.2] - 2026-02-03

### Fixed

- Replace git dependency (`strands-sglang @ git+...`) with PyPI package (`strands-sglang>=0.1.2`) to fix PyPI upload rejection.

## [0.0.1] - 2026-02-03 [yanked]

Initial release — core abstractions only. Environments will be added in future releases.

### Added

- **`Environment`** base class: `step()`, `reset()`, `cleanup()`, `get_tools()`, `get_hooks()`, `compute_metrics()`.
- **`Action` / `TaskContext`**: User message + ground truth, conversation history, and arbitrary metadata (`extra="allow"`).
- **`Observation`**: Step messages, metrics, and optional `TokenObservation` for TITO training.
- **`StepResult`**: Bundles observation, reward, and termination reason.
- **`TerminationReason`**: Maps agent exceptions (`MaxToolIterationsReachedError`, `MaxTokensReachedException`, timeouts) to enum values via cause-chain walking.
- **`RewardFunction` / `RewardResult`**: Abstract reward interface with scalar reward + diagnostics.
- **`ModelFactory`** type and factory functions for SGLang, Bedrock, and OpenAI backends.
- **`examples/math_env.py`**: Calculator tool example with exact-match reward, supporting SGLang and Bedrock.
- CI/CD: GitHub Actions for testing (lint + unit tests on Python 3.10–3.12) and PyPI publishing.
