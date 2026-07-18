# Why `dp8a` → `dp4a` for `ptxas` (and why the sim still runs `dp8a`)

GPGPU-Sim / Accel-Sim uses the same PTX in **two different ways**. Only one of them rewrites `dp8a`.

## 1. NVIDIA `ptxas` (occupancy / register count only)

GPGPU-Sim runs something like:

```text
ptxas -v foo.ptx  →  foo.ptxas   # “used N registers…”
```

It **never executes** that PTX. It only wants resource info (registers, spills) for occupancy / CTA limits.

Stock NVIDIA `ptxas` does **not** know the custom opcode `dp8a`, so it fails (often reported as error `65280` / “Ensure ptxas is in your path”).

**Workaround:** feed `ptxas` a **temporary copy** where:

```text
dp8a  →  dp4a
```

- Same instruction arity (4 register operands) → register count is a decent stand-in
- That temp file is **discarded** after `ptxas` finishes
- The original `foo.ptx` is **not** modified

This is what `ptxas_dp8a_wrap.sh` / `cuda_dp8a_shim` does (and what the Accel-Sim `ptx_loader.cc` patch does in-tree).

## 2. GPGPU-Sim functional / timing simulation (the real run)

Separately, Accel-Sim **parses the original** `foo.ptx` with its own lexer/parser (`ptx.l` + `dp8a_impl`). That file still contains:

```ptx
dp8a.s32.s32 %r1, %r2, %r3, %r0;
```

So the simulator executes **`dp8a`** (8× int4 MACs + accumulate), **not** `dp4a`.

```text
                 original PTX (has dp8a)
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
   temp: dp8a → dp4a            GPGPU-Sim parser
          ▼                           ▼
      NVIDIA ptxas              execute dp8a_impl
   (regs / occupancy only)      (int4×int4 timing)
```

## Short version

The rewrite exists **only** so the external `ptxas` tool can succeed.  
The simulator **never runs** the rewritten file; it always executes the original PTX with `dp8a`.

## Related naming note

| opcode | element width | lanes in a 32-bit word | meaning |
|--------|---------------|------------------------|---------|
| `dp4a` (NVIDIA) | int8 | 4 | 4× int8 MAC |
| `dp8a` (ours) | int4 | 8 | 8× int4 MAC |

The digit is **lane count**, not bit width.
