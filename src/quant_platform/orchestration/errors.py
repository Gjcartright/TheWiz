from __future__ import annotations


class OrchestratorError(RuntimeError):
    """Base error for local orchestrator failures."""


class UnknownStageError(OrchestratorError):
    """Raised when a requested orchestrator stage is not registered."""
