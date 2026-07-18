#!/usr/bin/env python3
"""Graft Hopper (sm_90) control bits into a trace's enhanced_execution_info.json.

Traces captured from a dlopen'd CUDA lib (e.g. llama.cpp's libggml-cuda.so) go through
the NVBit-SASS fallback: opcodes + operands are present but the control word is not, so
control_bits are all zero and the simulator falls back to the register scoreboard
(use_traditional_scoreboarding = !is_captured_from_binary, see sm.cc). This tool back-fills
the control bits from an offline `cuobjdump -sass` dump of the *matching* sm_90 .so, so the
simulator can use the real Hopper stall-count + wait-barrier dependency model.

It never touches the dynamic per-CTA .pb trace -- only the static table -- and only flips a
kernel's is_captured_from_binary to true when EVERY instruction validates (same opcode at
the same PC as the NVBit-captured trace). Kernels that don't fully match are left as
fallback, so you never get the dangerous "flag set but bits zero" state.

Decode mirrors control_bits.cc exactly (CCPos_arch_7x_8x = 41):
    cb = int(ctrl_word_hex, 16) >> 41
    stall_count          = cb        & 0xf
    is_yield             = not ((cb >> 4)  & 1)
    id_new_write_barrier = (cb >> 5) & 7   ; is_new_write_barrier = id != 7
    id_new_read_barrier  = (cb >> 8) & 7   ; is_new_read_barrier  = id != 7
    wait_barrier_bits    = (cb >> 11) & 0x3f

Produce the SASS dump on a machine with CUDA >= 12 (sm_90 needs it):
    cuobjdump -sass /path/to/libggml-cuda.so > ggml_sm90.sass

Usage:
    # validate only (prints per-kernel opcode-match coverage; writes nothing)
    graft_ctrl.py --sass ggml_sm90.sass --json .../extra_info/enhanced_execution_info.json \
                  --check [--funcid 339 --funcid 362]

    # graft: write an enriched copy (point a parallel trace dir's extra_info at it)
    graft_ctrl.py --sass ggml_sm90.sass --json .../extra_info/enhanced_execution_info.json \
                  --out  .../decode_traces_ctrl/extra_info/enhanced_execution_info.json

Tip for a non-destructive run: make a parallel trace dir that symlinks the big files and
holds the enriched extra_info, then point -trace at it:
    mkdir -p decode_traces_ctrl/extra_info
    ln -sf  decode_traces/dynamic_trace.pb  decode_traces_ctrl/dynamic_trace.pb
    ln -sf  decode_traces/threadblocks      decode_traces_ctrl/threadblocks
"""
import argparse
import json
import re
import sys

SHIFT = 41  # CCPos_arch_7x_8x (Volta..Hopper), see traced_constants.h

# /*PC*/ [optional @P / @!P / @UP predicate] OPCODE ...
INSN_RE = re.compile(r"/\*([0-9a-fA-F]+)\*/\s*(?:@!?U?P[T0-9]+\s+)?([^\s;]+)")
HEX_RE = re.compile(r"/\*\s*(0x[0-9a-fA-F]+)\s*\*/")


def strip_variant(name):
    """Drop the simulator's '___<variant_id>' suffix to recover the real mangled symbol."""
    return re.sub(r"___\d+$", "", name)


def decode_ctrl(ctrl_hex):
    cb = int(ctrl_hex, 16) >> SHIFT
    idw = (cb >> 5) & 7
    idr = (cb >> 8) & 7
    return {
        "stall_count": cb & 0xF,
        "is_yield": not bool((cb >> 4) & 1),
        "is_new_write_barrier": idw != 7,
        "id_new_write_barrier": idw,
        "is_new_read_barrier": idr != 7,
        "id_new_read_barrier": idr,
        "wait_barrier_bits": (cb >> 11) & 0x3F,
    }


def parse_sass(sass_path, wanted):
    """Stream the (large) SASS dump once; return {bare_name: {pc:int -> (op, inst_hex, ctrl_hex)}}
    for only the functions in `wanted`."""
    funcs = {}
    cur = None
    cur_pc = cur_op = cur_inst_hex = None
    with open(sass_path, "r", errors="replace") as f:
        for line in f:
            if "Function :" in line:
                name = line.split("Function :")[1].strip()
                cur = funcs.setdefault(name, {}) if name in wanted else None
                cur_pc = None
                continue
            if cur is None:
                continue
            m = INSN_RE.search(line)
            if m and "/*" in line and m.group(2)[0] != "0":
                # instruction line: pc, opcode, last hex on line = instruction word
                cur_pc = int(m.group(1), 16)
                cur_op = m.group(2)
                hx = HEX_RE.findall(line)
                cur_inst_hex = hx[-1] if hx else None
                continue
            if cur_pc is not None:
                # the lone control-word line that follows the instruction
                hx = HEX_RE.findall(line)
                if hx:
                    cur[cur_pc] = (cur_op, cur_inst_hex, hx[-1])
                    cur_pc = None
    return funcs


def main():
    ap = argparse.ArgumentParser(description="Graft sm_90 control bits into a trace JSON.")
    ap.add_argument("--sass", required=True, help="cuobjdump -sass dump of the matching sm_90 .so")
    ap.add_argument("--json", required=True, help="trace extra_info/enhanced_execution_info.json")
    ap.add_argument("--out", help="write enriched JSON here (omit with --check)")
    ap.add_argument("--check", action="store_true", help="validate + report only; write nothing")
    ap.add_argument("--funcid", type=int, action="append", default=[],
                    help="with --check, restrict the printed report to these unique_function_id(s)")
    args = ap.parse_args()

    if not args.check and not args.out:
        ap.error("provide --out to write, or --check to validate only")

    d = json.load(open(args.json))
    kernels = d["kernels"]
    wanted = set(strip_variant(k["kernel_name"]) for k in kernels)
    print(f"[*] {len(kernels)} kernels, {len(wanted)} unique funcs. Parsing SASS ...", flush=True)
    funcs = parse_sass(args.sass, wanted)
    print(f"[*] SASS provided {len(funcs)}/{len(wanted)} wanted functions", flush=True)

    if args.check:
        for k in kernels:
            if args.funcid and k.get("unique_function_id") not in args.funcid:
                continue
            bare = strip_variant(k["kernel_name"])
            sass = funcs.get(bare, {})
            insns = k["instructions"]
            match = mism = nopc = 0
            for ins in insns:
                pc, op = ins["pc_num_dec"], ins["op_code"]
                if pc not in sass:
                    nopc += 1
                elif sass[pc][0] != op:
                    mism += 1
                else:
                    match += 1
            flag = "OK" if (mism == 0 and nopc == 0 and sass) else "PARTIAL/MISSING"
            print(f"    funcid={k.get('unique_function_id')} {bare[:42]:42} "
                  f"instrs={len(insns)} matched={match} mism={mism} missing={nopc} [{flag}]")
        return

    ok = fail = 0
    fails = []
    for k in kernels:
        bare = strip_variant(k["kernel_name"])
        sass = funcs.get(bare)
        insns = k["instructions"]
        good = bool(sass) and all(
            ins["pc_num_dec"] in sass and sass[ins["pc_num_dec"]][0] == ins["op_code"]
            for ins in insns
        )
        if not good:
            fail += 1
            fails.append(bare[:48])
            continue
        for ins in insns:
            _op, ihex, chex = sass[ins["pc_num_dec"]]
            ins["control_bits"] = decode_ctrl(chex)
            ins["encoded_instruction"] = [ihex or "0x0", chex]
        k["is_captured_from_binary"] = True
        ok += 1

    print(f"[*] grafted {ok} kernels; {fail} left as fallback (no/!match SASS)")
    for x in fails[:25]:
        print("    fallback:", x)
    json.dump(d, open(args.out, "w"))
    print(f"[*] wrote {args.out}")


if __name__ == "__main__":
    main()
