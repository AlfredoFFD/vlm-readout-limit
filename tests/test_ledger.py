"""Tests for the experiment ledger (autoresearch-pattern provenance log)."""

from __future__ import annotations

from pathlib import Path

from vwm.experiments.ledger import (
    HEADER,
    LedgerRow,
    append,
    read_ledger,
    short_commit,
    now_iso,
)


def test_short_commit_returns_string():
    sha = short_commit()
    assert isinstance(sha, str)
    assert len(sha) >= 4                       # 7-char short sha or "nogit"


def test_now_iso_is_utc_iso_format():
    s = now_iso()
    assert "T" in s
    assert s.endswith(("+00:00", "Z"))


def test_append_writes_header_once_and_appends(tmp_path: Path):
    p = tmp_path / "ledger.tsv"
    r1 = LedgerRow(
        commit="abc1234", timestamp=now_iso(),
        provider="openai", model="gpt-4o", condition="C1",
        n_trials=100, mean_acc=0.83, k_at_8=2.5, fail_rate=0.0,
        duration_s=42.0, status="keep", description="baseline run",
    )
    r2 = LedgerRow(
        commit="def5678", timestamp=now_iso(),
        provider="anthropic", model="claude-sonnet-4-6", condition="C2",
        n_trials=100, mean_acc=0.75, k_at_8=2.0, fail_rate=0.01,
        duration_s=55.5, status="keep", description="C2 with sequential prompt",
    )
    append(r1, p)
    append(r2, p)

    text = p.read_text(encoding="utf-8")
    lines = text.strip().split("\n")
    assert lines[0] == HEADER
    assert len(lines) == 3                     # header + 2 rows
    # Tab-separated count
    assert lines[1].count("\t") == HEADER.count("\t")


def test_round_trip(tmp_path: Path):
    p = tmp_path / "ledger.tsv"
    rows_in = [
        LedgerRow(
            commit="abc1234", timestamp=now_iso(),
            provider="openai", model="gpt-4o", condition="C1",
            n_trials=600, mean_acc=0.823456, k_at_8=2.74, fail_rate=0.0033,
            duration_s=900.1, status="keep", description="full C1 run",
        ),
        LedgerRow(
            commit="def5678", timestamp=now_iso(),
            provider="qwen3-vl-4b", model="qwen3-vl-4b", condition="C3",
            n_trials=600, mean_acc=0.51, k_at_8=0.1, fail_rate=0.02,
            duration_s=3600.0, status="keep", description="KV-truncated, near chance as expected",
        ),
    ]
    for r in rows_in:
        append(r, p)
    rows_out = read_ledger(p)
    assert len(rows_out) == 2
    for a, b in zip(rows_in, rows_out):
        assert a.commit == b.commit
        assert a.provider == b.provider
        assert a.condition == b.condition
        assert a.n_trials == b.n_trials
        assert abs(a.mean_acc - b.mean_acc) < 1e-6
        assert a.status == b.status


def test_description_escapes_tabs_and_newlines(tmp_path: Path):
    p = tmp_path / "ledger.tsv"
    r = LedgerRow(
        commit="x", timestamp=now_iso(),
        provider="o", model="m", condition="C1",
        n_trials=1, mean_acc=0.5, k_at_8=0.0, fail_rate=0.0,
        duration_s=1.0, status="keep",
        description="line1\twith tabs\nand newlines",
    )
    append(r, p)
    rows = read_ledger(p)
    assert len(rows) == 1
    assert "\n" not in rows[0].description
    assert "\t" not in rows[0].description
