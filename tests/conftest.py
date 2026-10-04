"""Shared mocks for the tests.

These tests are designed to catch the silent-failure modes that propagate
through several stages before surfacing as a confusing downstream error.
They run without API access (sandbox tests use inspect_ai's local sandbox
plus the mockllm model).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from inspect_ai.scorer import Score
from inspect_ai.solver import TaskState


@dataclass
class _MockSample:
    """Minimal stand-in for an inspect_ai EvalSample."""
    metadata: dict = field(default_factory=dict)
    score: Optional[Score] = None


@dataclass
class _MockLog:
    """Minimal stand-in for an inspect_ai EvalLog."""
    samples: list = field(default_factory=list)


def _make_state(metadata: dict, output_text: str = "") -> TaskState:
    """Build a TaskState lightweight enough for solver/scorer unit tests."""
    from inspect_ai.model import ChatMessageAssistant, ModelOutput

    state = TaskState(
        model="mockllm/model",
        sample_id="0",
        epoch=0,
        input="",
        messages=[],
    )
    state.metadata = dict(metadata)
    if output_text:
        state.output = ModelOutput.from_content(
            model="mockllm/model", content=output_text
        )
    return state


def _stub_test_code(scripted: dict):
    """Build a stub test_code that returns pre-baked ExecResult-like objects keyed by input.

    `scripted` maps repr(input) -> (success: bool, stdout: str, stderr: str).
    """
    @dataclass
    class _Exec:
        success: bool
        stdout: str
        stderr: str = ""
        returncode: int = 0

    async def stub(state, code, input, USING_DOCKER=False):
        key = repr(input)
        if key not in scripted:
            raise AssertionError(
                f"test_code stub got unexpected input {key}. "
                f"Scripted inputs: {list(scripted.keys())}"
            )
        success, stdout, stderr = scripted[key]
        return _Exec(success=success, stdout=stdout, stderr=stderr,
                     returncode=0 if success else 1)
    return stub
