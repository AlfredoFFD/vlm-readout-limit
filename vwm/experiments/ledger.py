"""Experiment ledger — the autoresearch-style TSV log of every run we make.

Borrowed pattern (not code) from `karpathy/autoresearch`:
  - one tab-separated row per experiment, append-only
  - `commit` hash makes runs reproducible from git
  - `status` is keep / discard / crash (the agent's verdict, applied after eval)
  - `description` is a free-form summary of what changed
  - the file is NOT committed (it's the agent's working log, not the source of truth);
    the source of truth is the git history + the JSONL per-trial outputs.

Schema:
  commit | timestamp | provider | model | condition | n_trials | mean_acc | k_at_8 |
  fail_rate | duration_s | status | description

Status semantics in our project:
  keep    -- this run produced a publishable result (e.g. it ran to completion with
             <5% failure rate and a sensible accuracy curve)
  discard -- this run completed but the parameters drifted away from the canonical
             paradigm or produced a malformed metric; not used in figures
  crash   -- the run errored before producing any K estimate

The pattern is most useful for the *prompt-engineering* and *local-model 4-bit setup*
phases, where we genuinely want to iterate fast on parameters while preserving a clean
audit trail. We do NOT use it for the proprietary API runs because each iteration costs
real dollars; for those, we make one careful, deterministic run per (model, condition).
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

LEDGER_PATH_DEFAULT = Path("results/ledger.tsv")
HEADER = (
    "commit\ttimestamp\tprovider\tmodel\tcondition\tn_trials\t"
    "mean_acc\tk_at_8\tfail_rate\tduration_s\tstatus\tdescription"
)


@dataclass
class LedgerRow:
    commit: str
    timestamp: str
    provider: str
    model: str
    condition: str
    n_trials: int
    mean_acc: float
    k_at_8: float
    fail_rate: float
    duration_s: float
    status: Literal["keep", "discard", "crash"]
    description: str

    def to_tsv(self) -> str:
        # Escape tabs/newlines in the free-form description.
        desc = self.description.replace("\t", " ").replace("\n", " ").strip()
        return (
            f"{self.commit}\t{self.timestamp}\t{self.provider}\t{self.model}\t"
            f"{self.condition}\t{self.n_trials}\t{self.mean_acc:.6f}\t{self.k_at_8:.4f}\t"
            f"{self.fail_rate:.4f}\t{self.duration_s:.1f}\t{self.status}\t{desc}"
        )


def short_commit() -> str:
    """Current git HEAD short-sha, or 'nogit' if not in a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip()
    except Exception:                                  # noqa: BLE001
        return "nogit"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append(row: LedgerRow, path: Path | str = LEDGER_PATH_DEFAULT) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists()
    with p.open("a", encoding="utf-8") as f:
        if write_header:
            f.write(HEADER + "\n")
        f.write(row.to_tsv() + "\n")
    return p


def read_ledger(path: Path | str = LEDGER_PATH_DEFAULT) -> list[LedgerRow]:
    p = Path(path)
    rows: list[LedgerRow] = []
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 12:
                continue
            rows.append(LedgerRow(
                commit=parts[0], timestamp=parts[1],
                provider=parts[2], model=parts[3], condition=parts[4],
                n_trials=int(parts[5]),
                mean_acc=float(parts[6]),
                k_at_8=float(parts[7]),
                fail_rate=float(parts[8]),
                duration_s=float(parts[9]),
                status=parts[10],                                  # type: ignore[arg-type]
                description=parts[11],
            ))
    return rows
