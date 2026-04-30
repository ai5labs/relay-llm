"""Shared fixtures for contract tests."""

from __future__ import annotations

import os

import pytest


def _has(*names: str) -> bool:
    return all(os.environ.get(n) for n in names)


def skip_unless_keys(*names: str) -> pytest.MarkDecorator:
    """Skip the marked test unless all listed env vars are set."""
    return pytest.mark.skipif(
        not _has(*names),
        reason=f"requires env: {', '.join(names)}",
    )
