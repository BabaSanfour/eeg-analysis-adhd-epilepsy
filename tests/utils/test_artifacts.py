"""Shared persistence helpers, `utils/artifacts.py`."""

from __future__ import annotations

from eeg_adhd_epilepsy.utils.artifacts import write_text_atomic


def test_atomic_write_gets_the_mode_of_any_new_file(tmp_path):
    """The group can read what write_text_atomic writes, as it can any other new file."""
    plain = tmp_path / "plain.txt"
    plain.write_text("x\n")
    atomic = tmp_path / "atomic.txt"
    write_text_atomic(atomic, "x\n")

    assert atomic.stat().st_mode == plain.stat().st_mode
    assert atomic.read_text() == "x\n"
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []
