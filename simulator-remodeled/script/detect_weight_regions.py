#!/usr/bin/env python3
"""Detect the weight-matrix DRAM address region for one or more kernel launches,
purely from the trace's own per-CTA recorded addresses (no SASS/binary needed).

Heuristic: sample CTAs spread across the kernel's grid, group warp-0 addresses by PC.
A PC is "weight-like" if every sampled CTA produced a *distinct* address (each CTA/row
reads a different slice of one large buffer), the address isn't in this simulator's
shared-memory address family (0x32000_0000-prefixed), AND the PC's static opcode (cross-
referenced from enhanced_execution_info.json) is a load, not a store -- needed because a
per-CTA-varying *write* address (e.g. the dst output row) passes the first two checks just
as well as a weight load does, and merging it in falsely inflates the detected span (this
bit by an early version of this script: a single misclassified STG.E PC turned a real
~18MB weight region into a bogus 15GB span by bridging the gap to an unrelated buffer).
The weight region is the [min,max] address span across all qualifying load PCs, padded out
to 128B cache-line boundaries.

Usage: python3 detect_weight_regions.py <dynamic_trace.pb> <kernel_id|start-end> [...]
  e.g. python3 detect_weight_regions.py trace.pb 2366 2370 2372
  e.g. python3 detect_weight_regions.py trace.pb 2364-2404
Kernels in the given id(s)/range that don't match the weight-load pattern (anything
that isn't a GEMM/GEMV-style kernel with a per-CTA-varying large weight read) are
printed as "no region detected" and simply skipped, so it's safe to pass a whole
kernel-id range spanning unrelated kernel types -- only the matching ones end up in
the output JSON.
Writes weight_regions.json next to this script (or pass OUT_PATH env var).
"""
import sys
import os
import json
import collections

sys.path.insert(0, "/tmp/pb_py")
import trace_pb2  # noqa: E402
import threadblock_pb2  # noqa: E402

CACHE_LINE = 128
N_SAMPLES = 32
MIN_WEIGHT_SPAN_BYTES = 1_000_000  # weight matrices are multi-MB; small per-CTA auxiliary
# reads (e.g. an `ids` gather-index array) can also be distinct-per-CTA loads but span only
# a few KB -- this threshold is what actually separates them empirically (~17.8MB vs ~16KB
# observed on kernel 2380's `ids`-enabled variant).

_opcode_cache = {}  # kernel_name -> {pc_num_dec: op_code}


def is_shared_mem_addr(addr):
    return (addr >> 32) == 0x3 or addr < 0x100000


def load_pc_opcodes(json_path, kernel_name):
    """Targeted extraction of one kernel's instructions->opcode map from the (huge)
    enhanced_execution_info.json, without loading the whole file as JSON."""
    if kernel_name in _opcode_cache:
        return _opcode_cache[kernel_name]
    needle = f'"kernel_name":"{kernel_name}"'
    with open(json_path, "r") as f:
        data = f.read()
    idx = data.find(needle)
    if idx == -1:
        _opcode_cache[kernel_name] = {}
        return {}
    # back up to the start of this kernel's enclosing '{'
    obj_start = data.rfind("{", 0, idx)
    # scan forward counting brace depth to find the matching close
    depth = 0
    end = obj_start
    for i in range(obj_start, len(data)):
        if data[i] == "{":
            depth += 1
        elif data[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    obj = json.loads(data[obj_start:end])
    opcodes = {inst["pc_num_dec"]: inst["op_code"] for inst in obj.get("instructions", [])}
    _opcode_cache[kernel_name] = opcodes
    return opcodes


def detect_region(trace_dir, kernel_global_id, grid_size, kernel_name, json_path):
    kdir = os.path.join(trace_dir, "threadblocks", "device_0", "stream_0",
                         f"kernel_{kernel_global_id}")
    if not os.path.isdir(kdir):
        return None

    n = min(N_SAMPLES, grid_size)
    step = max(1, grid_size // n)
    cta_ids = sorted(set(list(range(0, grid_size, step)) + [grid_size - 1]))

    pc_addrs = collections.defaultdict(dict)  # pc -> {cta_id: addr}
    sampled = 0
    for cta in cta_ids:
        path = os.path.join(kdir, f"d_0_s_0_k_{kernel_global_id}_{cta},0,0.pb")
        if not os.path.isfile(path):
            continue
        tb = threadblock_pb2.threadblock()
        with open(path, "rb") as f:
            tb.ParseFromString(f.read())
        if 0 not in tb.warps:
            continue
        sampled += 1
        for inst in tb.warps[0].instructions:
            for addr in inst.addresses:
                pc_addrs[inst.pc][cta] = addr.base_address

    if sampled < 2:
        return None

    opcodes = load_pc_opcodes(json_path, kernel_name)

    weight_pcs = []
    all_addrs = []
    for pc, by_cta in pc_addrs.items():
        addrs = list(by_cta.values())
        if len(set(addrs)) != len(addrs):
            continue  # not distinct per CTA -> broadcast/shared/constant, skip
        if any(is_shared_mem_addr(a) for a in addrs):
            continue
        if len(addrs) < sampled * 0.5:
            continue  # too few CTAs hit this PC at all (e.g. a tail-tile-only PC under
            # Stream-K scheduling, where not every CTA executes the same loop-iteration
            # count) -- relaxed from requiring *every* sampled CTA, which found nothing
            # for k3 (mul_mat_q) even though plenty of real weight-load PCs were present
            # at ~97% coverage (128/132 CTAs)
        op = opcodes.get(pc, "")
        if not op.startswith("LD"):
            continue  # e.g. STG.E (dst write) -- distinct-per-CTA too, but not a weight load
        if max(addrs) - min(addrs) < MIN_WEIGHT_SPAN_BYTES:
            continue  # e.g. a small `ids` gather-index read -- a real load, distinct-per-CTA,
            # but too small a footprint to be the weight matrix
        weight_pcs.append(pc)
        all_addrs.extend(addrs)

    if not all_addrs:
        return None

    base = (min(all_addrs) // CACHE_LINE) * CACHE_LINE
    end = ((max(all_addrs) // CACHE_LINE) + 1) * CACHE_LINE
    return {
        "kernel_id": kernel_global_id,
        "base": base,
        "size": end - base,
        "weight_pc_count": len(weight_pcs),
        "weight_pcs": sorted(weight_pcs),
        "sampled_ctas": sampled,
    }


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    trace_pb_path = sys.argv[1]
    target_kernel_ids = set()
    for tok in sys.argv[2:]:
        if "-" in tok:
            lo, hi = tok.split("-", 1)
            target_kernel_ids.update(range(int(lo), int(hi) + 1))
        else:
            target_kernel_ids.add(int(tok))
    trace_dir = os.path.dirname(trace_pb_path)
    json_path = os.path.join(trace_dir, "extra_info", "enhanced_execution_info.json")

    t = trace_pb2.Trace()
    with open(trace_pb_path, "rb") as f:
        t.ParseFromString(f.read())

    results = {}
    for _, dev in t.gpu_device.items():
        for _, stream in dev.streams.items():
            for k in stream.kernels:
                if k.id not in target_kernel_ids:
                    continue
                grid_size = k.grid_dim.x * k.grid_dim.y * k.grid_dim.z
                region = detect_region(trace_dir, k.id, grid_size, k.name, json_path)
                if region is None:
                    print(f"kernel {k.id} ({k.name[:50]}): no region detected "
                          f"(grid={grid_size})")
                    continue
                region["kernel_name"] = k.name
                results[str(k.id)] = region
                print(f"kernel {k.id} ({k.name[:50]}): base=0x{region['base']:x} "
                      f"size={region['size']} ({region['size']/1e6:.2f} MB) "
                      f"weight_pcs={region['weight_pc_count']} "
                      f"sampled={region['sampled_ctas']}/{grid_size} CTAs")

    out_path = os.environ.get("OUT_PATH",
                               os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "weight_regions.json"))
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
