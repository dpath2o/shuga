from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Iterator

from .paths import ShugaPaths
from .types import (
    ClassificationSpec,
    MetricsSpec,
    ObservationSpec,
    PlottingSpec,
    RunSpec,
)


@dataclass(slots=True)
class ShugaContext:
    """Lightweight current-session context for interactive notebooks."""

    run: RunSpec | None = None
    classify: ClassificationSpec | None = None
    metrics: MetricsSpec | None = None
    plotting: PlottingSpec | None = None
    observations: ObservationSpec | None = None
    paths: ShugaPaths | None = None
    chunks: dict | None = None
    logger: object | None = None


_CURRENT_CONTEXT: ContextVar[ShugaContext | None] = ContextVar(
    "shuga_current_context",
    default=None,
)


def get_current_context() -> ShugaContext:
    """Return the active shuga context, or an empty context if none is set."""

    return _CURRENT_CONTEXT.get() or ShugaContext()


def set_current_context(
    *,
    run: RunSpec | None = None,
    classify: ClassificationSpec | None = None,
    metrics: MetricsSpec | None = None,
    plotting: PlottingSpec | None = None,
    observations: ObservationSpec | None = None,
    paths: ShugaPaths | None = None,
    chunks: dict | None = None,
    logger: object | None = None,
) -> ShugaContext:
    """Set the current shuga context for subsequent contextual loader calls."""

    ctx = ShugaContext(
        run=run,
        classify=classify,
        metrics=metrics,
        plotting=plotting,
        observations=observations,
        paths=paths,
        chunks=chunks,
        logger=logger,
    )
    _CURRENT_CONTEXT.set(ctx)
    return ctx


def clear_current_context() -> None:
    """Clear the active shuga context."""

    _CURRENT_CONTEXT.set(None)


@contextmanager
def use_current_context(
    *,
    run: RunSpec | None = None,
    classify: ClassificationSpec | None = None,
    metrics: MetricsSpec | None = None,
    plotting: PlottingSpec | None = None,
    observations: ObservationSpec | None = None,
    paths: ShugaPaths | None = None,
    chunks: dict | None = None,
    logger: object | None = None,
) -> Iterator[ShugaContext]:
    """Temporarily set a shuga context within a with block."""

    ctx = ShugaContext(
        run=run,
        classify=classify,
        metrics=metrics,
        plotting=plotting,
        observations=observations,
        paths=paths,
        chunks=chunks,
        logger=logger,
    )
    token: Token[ShugaContext | None] = _CURRENT_CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _CURRENT_CONTEXT.reset(token)
