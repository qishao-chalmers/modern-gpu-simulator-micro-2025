#!/usr/bin/env python3
"""Analyze [commit_trace] lines: op/opcode inventory + duration distribution per op,
to check whether duration is appropriately dynamic (e.g. memory ops showing a
cache-hit vs DRAM-miss bimodal split) or suspiciously constant.

Usage: python3 analyze_commit_trace.py <log_file>
"""
import re
import sys
import statistics
import collections

LINE_RE = re.compile(
    r'\[commit_trace\] sm=(\d+) subcore=(\d+) warp=(\d+) pc=(0x[0-9a-fA-F]+) '
    r'op=(\S+) opcode=(\S*) operands=(.*?) '
    r'issue_cycle=(\d+) commit_cycle=(\d+) duration=(\d+)'
)

def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * p))
    return sorted_vals[idx]

def main(path):
    by_op = collections.defaultdict(list)
    by_op_opcode = collections.defaultdict(list)
    op_opcode_counts = collections.Counter()
    total = 0
    skipped = 0

    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                skipped += 1
                continue
            total += 1
            op = m.group(5)
            opcode = m.group(6)
            dur = int(m.group(10))
            by_op[op].append(dur)
            by_op_opcode[(op, opcode)].append(dur)
            op_opcode_counts[(op, opcode)] += 1

    print(f"# total parsed={total} skipped={skipped}\n")

    print("## Op-type inventory (count, duration stats)")
    print(f"{'op':<30}{'count':>10}{'distinct_dur':>14}{'min':>8}{'p50':>8}{'mean':>8}{'p90':>8}{'p99':>8}{'max':>8}")
    for op, durs in sorted(by_op.items(), key=lambda kv: -len(kv[1])):
        s = sorted(durs)
        distinct = len(set(durs))
        mean = statistics.mean(durs)
        print(f"{op:<30}{len(durs):>10}{distinct:>14}{s[0]:>8}{percentile(s,0.5):>8}"
              f"{mean:>8.1f}{percentile(s,0.9):>8}{percentile(s,0.99):>8}{s[-1]:>8}")

    print("\n## Per-opcode breakdown for memory ops (LOAD_OP / STORE_OP) -- looking for bimodal hit-vs-miss latency")
    for op in ("LOAD_OP", "STORE_OP", "MEMORY_MISCELLANEOUS_OP"):
        opcodes_for_op = [(oc, durs) for (o, oc), durs in by_op_opcode.items() if o == op]
        if not opcodes_for_op:
            continue
        print(f"\n--- {op} ---")
        for opcode, durs in sorted(opcodes_for_op, key=lambda kv: -len(kv[1])):
            s = sorted(durs)
            distinct = len(set(durs))
            # crude histogram buckets to spot bimodality (cache hit vs DRAM RTT)
            buckets = collections.Counter()
            for d in durs:
                if d < 20:
                    b = "0-20"
                elif d < 50:
                    b = "20-50"
                elif d < 100:
                    b = "50-100"
                elif d < 200:
                    b = "100-200"
                elif d < 300:
                    b = "200-300"
                elif d < 500:
                    b = "300-500"
                else:
                    b = "500+"
                buckets[b] += 1
            hist_str = " ".join(f"{b}:{buckets[b]}" for b in
                                 ["0-20","20-50","50-100","100-200","200-300","300-500","500+"] if buckets[b])
            print(f"  {opcode or '(none)':<20} n={len(durs):<7} distinct_dur={distinct:<5} "
                  f"min={s[0]:<6} p50={percentile(s,0.5):<6} mean={statistics.mean(durs):<8.1f} "
                  f"p99={percentile(s,0.99):<6} max={s[-1]:<6}  hist=[{hist_str}]")

    print("\n## Per-opcode breakdown for compute ops (SP_OP / INTP_OP / SFU_OP / UNIFORM_OP) -- looking for unexpected variance")
    for op in ("SP_OP", "INTP_OP", "SFU_OP", "UNIFORM_OP", "TENSOR_CORE_OP"):
        opcodes_for_op = [(oc, durs) for (o, oc), durs in by_op_opcode.items() if o == op]
        if not opcodes_for_op:
            continue
        print(f"\n--- {op} ---")
        for opcode, durs in sorted(opcodes_for_op, key=lambda kv: -len(kv[1]))[:15]:
            s = sorted(durs)
            distinct = len(set(durs))
            print(f"  {opcode or '(none)':<20} n={len(durs):<7} distinct_dur={distinct:<5} "
                  f"min={s[0]:<6} p50={percentile(s,0.5):<6} mean={statistics.mean(durs):<8.1f} "
                  f"max={s[-1]:<6}")

if __name__ == "__main__":
    main(sys.argv[1])
