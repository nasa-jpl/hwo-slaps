"""CPU-only regression tests for supervised Fisher worker execution."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("autolens")

from hwoslaps.modeling.fisher_detector import _supervised_ordered_map


def _return_index(index):
    return index


def _die_on_one(index):
    if index == 1:
        os._exit(37)
    return index


def _set_child_environment():
    os.environ["HWOSLAPS_TEST_CHILD_ONLY"] = "child"


def test_dead_fisher_worker_surfaces_an_exception():
    """A worker exit must fail the ordered map instead of hanging."""
    with pytest.raises(Exception):
        list(_supervised_ordered_map(_die_on_one, range(8), num_workers=2))


def test_worker_initializer_does_not_mutate_parent_environment(monkeypatch):
    """Thread-limit setup belongs to child initializers, not the parent."""
    monkeypatch.setenv("HWOSLAPS_TEST_CHILD_ONLY", "parent")
    list(
        _supervised_ordered_map(
            _return_index,
            range(3),
            num_workers=2,
            initializer=_set_child_environment,
        )
    )
    assert os.environ["HWOSLAPS_TEST_CHILD_ONLY"] == "parent"


def test_supervised_ordered_map_preserves_input_order():
    assert list(_supervised_ordered_map(_return_index, [3, 1, 2], num_workers=2)) == [
        3,
        1,
        2,
    ]
