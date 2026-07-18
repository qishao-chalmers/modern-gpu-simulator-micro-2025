#!/usr/bin/env python3
"""
Fast selective control-bit enrichment for enhanced protobuf traces.

Uses cuobjdump -fun to dump SASS (+ hex encoding) for only the CUDA device
functions you care about, then patches extra_info/enhanced_execution_info.json
with encoded_instruction + control_bits and sets is_captured_from_binary=true.

Typical workflow (avoid parsing all 129 libggml-cuda cubins at trace shutdown):

  # 1) Trace with NVBit only (fast shutdown metadata)
  export TRACER_SKIP_CUBIN_DUMP=1
  LD_PRELOAD=./tracer_tool.so llama-bench ...

  # 2) Patch control bits for kernels you simulate
  ./patch_control_bits_from_cubin.py patch \\
    --cuda-binary /path/to/libggml-cuda.so \\
    --json traces/extra_info/enhanced_execution_info.json \\
    --match rope_neox,set_rows

  # Discover mangled names first
  ./patch_control_bits_from_cubin.py list \\
    --cuda-binary /path/to/libggml-cuda.so --match rope
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

VARIANT_DELIMITER = "___"
CC_POS_ARCH_7X_8X = 41
PT_PREDICATE_REGISTER = 7  # traced_constants.h PT


@dataclass
class ControlBits:
    stall_count: int = 0
    is_yield: bool = False
    is_new_read_barrier: bool = False
    is_new_write_barrier: bool = False
    id_new_read_barrier: int = 0
    id_new_write_barrier: int = 0
    wait_barrier_bits: int = 0

    def to_json(self) -> dict:
        return {
            "stall_count": self.stall_count,
            "is_yield": self.is_yield,
            "is_new_read_barrier": self.is_new_read_barrier,
            "is_new_write_barrier": self.is_new_write_barrier,
            "id_new_read_barrier": self.id_new_read_barrier,
            "id_new_write_barrier": self.id_new_write_barrier,
            "wait_barrier_bits": self.wait_barrier_bits,
        }


@dataclass
class ParsedInstruction:
    pc_num: int
    pc_string_hex: str
    op_code: str
    encoded_instruction: List[str]
    control_bits: ControlBits
    is_predicated: bool = False
    is_uniform_predicate: bool = False
    is_predicate_negate: bool = False
    predicate_register: int = 0
    operand_tokens: List[str] = field(default_factory=list)


def decode_control_bits(encoded_word2: str) -> ControlBits:
    """Mirror control_bits.cc (Volta+ / Hopper, shift=41)."""
    n = int(encoded_word2, 16)
    cb = (n >> CC_POS_ARCH_7X_8X) & 0xFFFFFFFF
    id_w = (cb & 0xE0) >> 5
    id_r = (cb & 0x700) >> 8
    return ControlBits(
        stall_count=(cb & 0xF),
        is_yield=not bool((cb & 0x10) >> 4),
        id_new_write_barrier=id_w,
        id_new_read_barrier=id_r,
        wait_barrier_bits=(cb & 0x1F800) >> 11,
        is_new_write_barrier=(id_w != 7),
        is_new_read_barrier=(id_r != 7),
    )


def strip_sass_line(line: str) -> str:
    line = line.replace("/*", " ").replace("*/", " ")
    line = line.replace(",", " ").replace(";", " ")
    return " ".join(line.split())


def strip_sass_annotations(line: str) -> str:
    line = re.sub(r"\s*&req=\{[^}]+\}", "", line)
    line = re.sub(r"\s*&wr=0x[0-9A-Fa-f]+", "", line)
    line = re.sub(r"\s*&rd=0x[0-9A-Fa-f]+", "", line)
    line = re.sub(r"\s*\?[A-Za-z0-9_]+", "", line)
    return line


def parse_predicate(tokens: List[str], idx: int) -> Tuple[bool, bool, bool, int, int]:
    if idx >= len(tokens) or not tokens[idx].startswith("@"):
        return False, False, False, 0, idx
    tok = tokens[idx]
    pos = 1
    negate = False
    uniform = False
    if pos < len(tok) and tok[pos] == "!":
        negate = True
        pos += 1
    if pos < len(tok) and tok[pos] == "U":
        uniform = True
        pos += 1
    if pos >= len(tok):
        return True, uniform, negate, 0, idx + 1
    pred_ch = tok[pos]
    if pred_ch == "T":
        pred_reg = PT_PREDICATE_REGISTER
    else:
        pred_reg = int(pred_ch, 10)
    return True, uniform, negate, pred_reg, idx + 1


def parse_instruction_line(
    part1: str, part2: Optional[str], arch_version: int
) -> Optional[ParsedInstruction]:
    part1 = strip_sass_annotations(part1)
    tokens = strip_sass_line(part1).split()
    if len(tokens) < 2:
        return None
    try:
        pc_num = int(tokens[0], 16)
    except ValueError:
        return None

    idx = 1
    is_pred, is_upred, is_neg, pred_reg, idx = parse_predicate(tokens, idx)
    if idx >= len(tokens):
        return None
    op_code = tokens[idx]
    idx += 1
    operand_tokens = [t for t in tokens[idx:] if t and t[0] not in "&?"]

    hex_part1 = [t for t in tokens if t.startswith("0x")]
    hex_part2: List[str] = []
    if part2 is not None:
        hex_part2 = [t for t in strip_sass_line(part2).split() if t.startswith("0x")]

    encoded: List[str] = []
    if arch_version >= 70:
        if len(hex_part2) >= 2:
            encoded = [hex_part2[0], hex_part2[1]]
        elif len(hex_part2) == 1 and hex_part1:
            encoded = [hex_part1[-1], hex_part2[0]]
        elif len(hex_part2) == 1:
            encoded = [hex_part2[0], hex_part2[0]]
        elif len(hex_part1) >= 2:
            encoded = [hex_part1[-2], hex_part1[-1]]
        elif len(hex_part1) == 1:
            encoded = [hex_part1[0], hex_part1[0]]
        else:
            return None
    else:
        if hex_part1:
            encoded = [hex_part1[-1]]
        elif hex_part2:
            encoded = [hex_part2[0]]
        else:
            return None

    control_word = encoded[1] if len(encoded) > 1 else encoded[0]
    return ParsedInstruction(
        pc_num=pc_num,
        pc_string_hex=f"{pc_num:x}",
        op_code=op_code,
        encoded_instruction=encoded,
        control_bits=decode_control_bits(control_word),
        is_predicated=is_pred,
        is_uniform_predicate=is_upred,
        is_predicate_negate=is_neg,
        predicate_register=pred_reg,
        operand_tokens=operand_tokens,
    )


def parse_cuobjdump_sass(text: str, arch_version: int = 90) -> Dict[str, Dict[int, ParsedInstruction]]:
    """Return {mangled_function_name: {pc: ParsedInstruction}}."""
    kernels: Dict[str, Dict[int, ParsedInstruction]] = {}
    current_name: Optional[str] = None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = strip_sass_line(raw)
        if "Function" in stripped and ":" in stripped:
            parts = stripped.split()
            # "Function : _Zfoo..."
            try:
                colon = parts.index(":")
            except ValueError:
                i += 1
                continue
            if colon + 1 < len(parts):
                current_name = parts[colon + 1]
                kernels.setdefault(current_name, {})
            i += 1
            continue
        if stripped.startswith(".........."):
            current_name = None
            i += 1
            continue
        if current_name is None or "headerflags" in stripped:
            i += 1
            continue
        # Opcode line: next line is usually the hex encoding (sm_70+).
        part2 = None
        if arch_version >= 70 and i + 1 < len(lines):
            nxt_raw = lines[i + 1]
            nxt_tokens = strip_sass_line(nxt_raw).split()
            if any(t.startswith("0x") for t in nxt_tokens):
                part2 = nxt_raw
                i += 1
        inst = parse_instruction_line(raw, part2, arch_version)
        if inst is not None:
            kernels[current_name][inst.pc_num] = inst
        i += 1
    return kernels


def run_cuobjdump(args: List[str], cuda_binary: Path) -> str:
    cmd = ["cuobjdump", *args, str(cuda_binary)]
    print(f"[patch_control_bits] running: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"cuobjdump failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.stdout + proc.stderr


def list_cuda_functions(cuda_binary: Path) -> List[str]:
    out = run_cuobjdump(["-ltext", "all"], cuda_binary)
    names: List[str] = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("text") or line.startswith("Function"):
            continue
        # nvcc fatbin listing: ".text._Zmangled_name" or bare symbol
        if line.startswith(".text."):
            names.append(line[len(".text.") :])
        elif line.startswith("_Z"):
            names.append(line.split()[0])
    return sorted(set(names))


def filter_names(names: Sequence[str], patterns: Sequence[str]) -> List[str]:
    if not patterns:
        return list(names)
    out = []
    for n in names:
        if any(p in n for p in patterns):
            out.append(n)
    return out


def kernel_base_name(kernel_json_name: str) -> str:
    if VARIANT_DELIMITER in kernel_json_name:
        return kernel_json_name.split(VARIANT_DELIMITER, 1)[0]
    return kernel_json_name


def chunk_names(names: Sequence[str], size: int) -> List[List[str]]:
    return [list(names[i : i + size]) for i in range(0, len(names), size)]


def dump_sass_for_functions(
    cuda_binary: Path, function_names: Sequence[str], arch: str = "sm_90"
) -> str:
    if not function_names:
        return ""
    combined = []
    for batch in chunk_names(function_names, 8):
        fun_arg = ",".join(batch)
        out = run_cuobjdump(["-sass", f"-arch={arch}", f"-fun={fun_arg}"], cuda_binary)
        combined.append(out)
    return "\n".join(combined)


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def patch_json_kernels(
    doc: dict,
    parsed: Dict[str, Dict[int, ParsedInstruction]],
    only_bases: Optional[Set[str]] = None,
) -> Tuple[int, int, List[str]]:
    """Patch kernels in enhanced_execution_info.json. Returns (kernels_patched, inst_patched, warnings)."""
    warnings: List[str] = []
    kernels_patched = 0
    inst_patched = 0
    for kernel in doc.get("kernels", []):
        kname = kernel.get("kernel_name", "")
        base = kernel_base_name(kname)
        if only_bases is not None and base not in only_bases:
            continue
        if base not in parsed:
            warnings.append(f"no cuobjdump SASS for JSON kernel base name: {base} ({kname})")
            continue
        by_pc = parsed[base]
        if not by_pc:
            warnings.append(f"empty SASS parse for {base}")
            continue
        matched = 0
        for inst in kernel.get("instructions", []):
            pc = inst.get("pc_num_dec")
            if pc is None:
                continue
            p = by_pc.get(int(pc))
            if p is None:
                continue
            inst["encoded_instruction"] = list(p.encoded_instruction)
            inst["control_bits"] = p.control_bits.to_json()
            matched += 1
        if matched == 0:
            warnings.append(f"kernel {kname}: 0 instruction PCs matched cubin SASS")
            continue
        kernel["is_captured_from_binary"] = True
        kernels_patched += 1
        inst_patched += matched
        print(
            f"[patch_control_bits] patched {kname}: {matched} instructions, "
            f"max_stall={max(by_pc[x].control_bits.stall_count for x in by_pc)}",
            file=sys.stderr,
        )
    return kernels_patched, inst_patched, warnings


def cmd_list(args: argparse.Namespace) -> int:
    names = list_cuda_functions(args.cuda_binary)
    names = filter_names(names, split_patterns(args.match))
    for n in names:
        print(n)
    print(f"# {len(names)} function(s)", file=sys.stderr)
    return 0


def cmd_dump_sass(args: argparse.Namespace) -> int:
    patterns = split_patterns(args.match)
    names = filter_names(list_cuda_functions(args.cuda_binary), patterns)
    if args.functions:
        names = sorted(set(names) | set(args.functions))
    if not names:
        print("No functions matched.", file=sys.stderr)
        return 1
    sass = dump_sass_for_functions(args.cuda_binary, names, args.arch)
    out = args.output or Path("-")
    if str(out) == "-":
        sys.stdout.write(sass)
    else:
        out.write_text(sass)
        print(f"Wrote {out} ({len(sass)} bytes)", file=sys.stderr)
    return 0


def cmd_patch(args: argparse.Namespace) -> int:
    doc = load_json(args.json)
    patterns = split_patterns(args.match)

    if args.functions:
        target_names = list(args.functions)
    else:
        # Prefer kernels already present in JSON when patching a trace artifact.
        json_bases = {kernel_base_name(k["kernel_name"]) for k in doc.get("kernels", [])}
        if patterns:
            target_names = [n for n in json_bases if any(p in n for p in patterns)]
            if not target_names:
                all_names = list_cuda_functions(args.cuda_binary)
                target_names = filter_names(all_names, patterns)
        else:
            target_names = sorted(json_bases)

    if not target_names:
        print("No target functions resolved. Use --match and/or --functions.", file=sys.stderr)
        return 1

    print(f"[patch_control_bits] dumping SASS for {len(target_names)} function(s)", file=sys.stderr)
    sass = dump_sass_for_functions(args.cuda_binary, target_names, args.arch)
    parsed = parse_cuobjdump_sass(sass, arch_version=args.sm_version)
    found = {k for k, v in parsed.items() if v}
    missing = [n for n in target_names if n not in found]
    for m in missing:
        print(f"[patch_control_bits] WARN: no SASS parsed for {m}", file=sys.stderr)

    only_bases = set(target_names) if args.functions else None
    if patterns and not args.functions:
        only_bases = set(target_names)

    kp, ip, warnings = patch_json_kernels(doc, parsed, only_bases=only_bases)
    for w in warnings:
        print(f"[patch_control_bits] WARN: {w}", file=sys.stderr)

    out = args.output or args.json
    save_json(out, doc)
    print(
        f"[patch_control_bits] done: {kp} kernel(s), {ip} instruction(s) -> {out}",
        file=sys.stderr,
    )
    return 0 if kp > 0 else 1


def split_patterns(match: Optional[str]) -> List[str]:
    if not match:
        return []
    return [p.strip() for p in match.split(",") if p.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Selective cuobjdump -fun SASS dump + control_bits JSON patch"
    )
    p.add_argument(
        "--cuda-binary",
        type=Path,
        required=True,
        help="Path to libggml-cuda.so (or TRACER_CUDA_BINARY)",
    )
    p.add_argument(
        "--arch",
        default="sm_90",
        help="GPU arch for cuobjdump -sass (default: sm_90)",
    )
    p.add_argument(
        "--sm-version",
        type=int,
        default=90,
        help="SM version for hex-line parsing (default: 90)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    list_p = sub.add_parser("list", help="List mangled device function names in .so")
    list_p.add_argument(
        "--match",
        help="Comma-separated substrings to filter (e.g. rope_neox,set_rows)",
    )
    list_p.set_defaults(func=cmd_list)

    dump_p = sub.add_parser("dump-sass", help="Dump SASS for matched functions")
    dump_p.add_argument("--match", help="Comma-separated substrings")
    dump_p.add_argument(
        "--functions",
        nargs="*",
        help="Exact mangled function names (in addition to --match)",
    )
    dump_p.add_argument("-o", "--output", type=Path, help="Output .sass file (default: stdout)")
    dump_p.set_defaults(func=cmd_dump_sass)

    patch_p = sub.add_parser(
        "patch",
        help="Patch enhanced_execution_info.json with control_bits from cubin",
    )
    patch_p.add_argument(
        "--json",
        type=Path,
        required=True,
        help="Path to extra_info/enhanced_execution_info.json",
    )
    patch_p.add_argument(
        "--match",
        help="Comma-separated substrings; default=all kernels in JSON",
    )
    patch_p.add_argument(
        "--functions",
        nargs="*",
        help="Exact mangled names (bypass JSON kernel list)",
    )
    patch_p.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output JSON (default: overwrite --json)",
    )
    patch_p.set_defaults(func=cmd_patch)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.cuda_binary.is_file():
        print(f"CUDA binary not found: {args.cuda_binary}", file=sys.stderr)
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
