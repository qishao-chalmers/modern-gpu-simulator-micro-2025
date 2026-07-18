#!/usr/bin/env python3
"""Parse subcore detailed scheduling log (SM0 / subcore0 / dynamic warp 0).

Expects lines from subcore.cc debug prints like:
  <cycle> Instruction: LOAD_OP @ pc=0x00/0/0/ ready
  <cycle> Instruction: UNIFORM_OP @ pc=0x300/0/0/ not ready because: not yield: 1 ...

Usage:
  python3 parse_detailed.py [path/to/detailed.log]
  python3 parse_detailed.py   # defaults to detailed.log next to this script
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

FLAGS = [
    "not yield",
    "stall counter",
    "barriers",
    "fu_avail",
    "prog_barrier",
    "ldgdepbar",
    "scoreboards",
    "result_queue",
]


def parse_line_flags(line: str) -> list[str]:
    """Return flag names that are blocking (value 0)."""
    bad: list[str] = []
    for fl in FLAGS:
        m = re.search(rf"{re.escape(fl)}:\s*(\d+)", line)
        if m and m.group(1) == "0":
            bad.append(fl)
    return bad


def extract_inst(line: str) -> str | None:
    """Instruction class and PC from log line."""
    m = re.search(r"Instruction:\s+(\S+)\s+@ pc=([^/\s]+)", line)
    if not m:
        return None
    return f"{m.group(1)}@{m.group(2)}"


def parse(path: Path) -> int:
    lines = path.read_text(errors="replace").splitlines()

    ready = [l for l in lines if l.rstrip().endswith("ready")]
    not_ready = [l for l in lines if "not ready because" in l]

    print(f"log: {path}")
    print(f"total_lines: {len(lines)}")
    print(f"ready_events: {len(ready)}")
    print(f"not_ready_events: {len(not_ready)}")

    block: Counter[str] = Counter()
    sole: Counter[str] = Counter()
    combo: Counter[tuple[str, ...]] = Counter()
    inst_counter: Counter[str] = Counter()
    inst_block: dict[str, Counter[str]] = defaultdict(Counter)

    for line in not_ready:
        bad = parse_line_flags(line)
        for b in bad:
            block[b] += 1
        if len(bad) == 1:
            sole[bad[0]] += 1
        elif bad:
            combo[tuple(sorted(bad))] += 1

        inst = extract_inst(line)
        if inst:
            inst_counter[inst] += 1
            for b in bad:
                inst_block[inst][b] += 1

    n = len(not_ready) or 1
    print("\n=== Blocking flags (% of not-ready events) ===")
    for k, v in block.most_common():
        print(f"  {k:16s} {v:8d} ({100 * v / n:.1f}%)")

    print("\n=== Sole blockers ===")
    for k, v in sole.most_common():
        print(f"  {k:16s} {v:8d} ({100 * v / n:.1f}%)")

    if combo:
        print("\n=== Top multi-blocker combos ===")
        for c, v in combo.most_common(10):
            print(f"  {c}: {v}")

    by_cycle: dict[int, dict[str, int]] = defaultdict(
        lambda: {"ready": 0, "not_ready": 0}
    )
    cycles: list[int] = []
    for line in lines:
        m = re.match(r"^(\d+)\s+Instruction:", line)
        if not m:
            continue
        c = int(m.group(1))
        cycles.append(c)
        if "not ready because" in line:
            by_cycle[c]["not_ready"] += 1
        elif line.rstrip().endswith("ready"):
            by_cycle[c]["ready"] += 1

    if cycles:
        print("\n=== Cycle span ===")
        print(f"  min={min(cycles)} max={max(cycles)}")
        print(f"  logged_cycles={len(by_cycle)}")

    if by_cycle:
        only_nr = sum(
            1 for v in by_cycle.values() if v["not_ready"] and not v["ready"]
        )
        only_r = sum(
            1 for v in by_cycle.values() if v["ready"] and not v["not_ready"]
        )
        both = sum(
            1 for v in by_cycle.values() if v["ready"] and v["not_ready"]
        )
        total = len(by_cycle)
        print("\n=== Per-cycle issue gate ===")
        print(f"  only_not_ready: {only_nr} ({100 * only_nr / total:.1f}%)")
        print(f"  only_ready:     {only_r} ({100 * only_r / total:.1f}%)")
        print(f"  both_same_cycle:{both} ({100 * both / total:.1f}%)")

    if inst_counter:
        print("\n=== Top instructions when not ready ===")
        for inst, v in inst_counter.most_common(20):
            bs = ", ".join(f"{k}={n}" for k, n in inst_block[inst].most_common(3))
            print(f"  {v:6d}  {inst}  [{bs}]")

    print("\n=== Sample not-ready lines ===")
    for line in not_ready[:8]:
        print(f"  {line[:280]}")

    if ready:
        print("\n=== Sample ready lines ===")
        for line in ready[:5]:
            print(f"  {line[:280]}")

    return 0


def main() -> int:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = Path(__file__).resolve().parent / "detailed.log"

    if not path.is_file():
        print(f"ERROR: log not found: {path}", file=sys.stderr)
        return 1

    return parse(path)


if __name__ == "__main__":
    raise SystemExit(main())
