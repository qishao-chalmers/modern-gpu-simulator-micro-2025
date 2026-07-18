# Qwen3-14B decode: one full layer, kernel-by-kernel (real H100 vs simulator)

## Scope

One complete repeating decode-layer kernel sequence for Qwen3-14B-Q8_0 (`-p 1024 -n 64 -b 2048
-ngl 99 --flash-attn 1`), identified directly from the trace (kernel IDs **2628-2649**, confirmed by
grid size against known model dims: hidden_size=5120, num_attention_heads=40, num_kv_heads=8,
head_dim=128, FFN intermediate≈8704). All 40 layers are structurally identical (same kernel
sequence, same grids); this table is one representative layer.

**No separate embedding kernel exists in this region.** For single-token decode, looking up one
embedding row is a pointer offset, not a GPU kernel — it costs ~0 measured GPU time. The actual
once-per-decode-step kernel is the LM head (`mul_mat_vec_q`, grid=151936, vocab size), which sits at
the very end of the *previous* decode step (kernel 2627 in this trace), not at the start of this one.

## One complete decode layer

| Kernel ID | Name | Grid | Role | Real H100 (Nsight Systems) | Simulator (SM90_H100_l2norm_l1dnorm, post MSHR-merge fix) |
|---|---|---|---|---|---|
| 2628 | `rms_norm_f32<1024>` | 1 | attn_norm | 5.952 μs | _pending_ |
| 2629 | `quantize_q8_1` | 20 | quantize for QKV input | 2.048 μs | _pending_ |
| 2630 | `mul_mat_vec_q` | 5120 | Q proj | 20.256 μs | _pending_ |
| 2631 | `rms_norm_f32<256>` | 40 | Q-norm (per-head, 40 heads) | 2.304 μs | _pending_ |
| 2632 | `rope_neox` | 40 | RoPE on Q | 2.048 μs | _pending_ |
| 2633 | `quantize_q8_1` | 20 | quantize for K proj | 2.240 μs | _pending_ |
| 2634 | `mul_mat_vec_q` | 1024 | K proj (8 KV heads × 128) | 6.048 μs | _pending_ |
| 2635 | `quantize_q8_1` | 20 | quantize for V proj | 1.920 μs | _pending_ |
| 2636 | `mul_mat_vec_q` | 1024 | V proj | 6.112 μs | _pending_ |
| 2637 | `rms_norm_f32<256>` | 8 | K-norm (per-head, 8 KV heads) | 2.304 μs | _pending_ |
| 2638 | `rope_neox` | 8 | RoPE on K | 2.240 μs | _pending_ |
| 2639 | `k_set_rows` | 4 | write V into KV cache | 1.856 μs | _pending_ |
| 2640 | `cpy_scalar_contiguous` | 4 | attn scale init? (likely layer-0-only setup, not steady-state per-layer work) | *not separately listed in the real extract — see note below* | _pending_ |
| 2641 | `flash_attn_ext_vec` | 80 | attention (split-K, 40 heads × 2 splits) | 4.352 μs | 4.3154 μs (`A:192:8`, embedded natural-layer run) |
| 2642 | `flash_attn_combine_results` | 40 | combine split-K results | 2.176 μs | _pending_ |
| 2643 | `quantize_q8_1` | 20 | quantize attn output | 2.048 μs | _pending_ |
| 2644 | `mul_mat_vec_q` | 5120 | O proj (output projection) | 20.928 μs | _pending_ |
| 2645 | `rms_norm_f32<1024>` | 1 | ffn_norm | 5.632 μs | _pending_ |
| 2646 | `quantize_q8_1` | 20 | quantize for FFN input | 2.080 μs | _pending_ |
| 2647 | `mul_mat_vec_q` | 17408 | FFN gate+up (fused) proj | 124.096 μs | _pending_ |
| 2648 | `quantize_q8_1` | 68 | quantize FFN intermediate | 2.592 μs | _pending_ |
| 2649 | `mul_mat_vec_q` | 5120 | FFN down_proj | 63.808 μs | _pending_ |

**Total real GPU-kernel time per layer ≈ 283.04 μs** (sum of the 21 kernels with a real-extract
entry; `cpy_scalar_contiguous` is excluded from the sum since it has no measured value).

## Notes

- **`cpy_scalar_contiguous` (kernel 2640) has no separate entry in the real-machine extract, and may
  not be part of the steady repeating decode-layer body.** The trace shows it as a distinct kernel
  launch (grid=4) immediately after `k_set_rows`, but the Nsight Systems summary used to build the
  real-machine column (`real_machine_timing.txt`) goes straight from `k_set_rows` (1.856 μs) to
  `flash_attn_ext_vec` (4.352 μs) with nothing in between. Separately, llama.cpp kernel-mapping
  notes identify `cpy_scalar_contiguous<f32→f16>` as an attention-scale initialization kernel that
  is **layer-0 only**, not a repeated per-layer decode kernel. So this row is currently kept as
  "present in this captured trace window" rather than assumed to be part of the canonical 21-kernel
  repeating layer sequence. Needs a direct check against the raw Nsight `.sqlite`/`.nsys-rep` and,
  ideally, a second trace window from a later layer if an exact classification is needed.
- **FFN dominates the layer**: gate+up (124.1 μs) + down_proj (63.8 μs) = ~187.9 μs, ~66% of the
  ~283 μs total — far more than the entire attention stack (QKV proj + RoPE + attention + O-proj
  ≈ 25 μs combined, excluding norms/quantize).
- The kernel-ID-to-role mapping was derived directly from the trace's own grid sizes (not assumed),
  then cross-checked against llama.cpp kernel-mapping notes for the K/V-cache write path:
  `rope_neox` handles the K-cache write, while `k_set_rows` corresponds to the V-cache write.
  Primary trace source: `dynamic_trace.pb` kernel entries for IDs 2600-2680, cross-checked against
  known Qwen3-14B dimensions (hidden_size=5120, num_attention_heads=40, num_kv_heads=8,
  head_dim=128).

## Source data

- Real-machine per-kernel durations: `/home/qshao/Project/Fun/gpu_traces/qwen14b/real_machine_timing.txt`
  (Nsight Systems extract, one clean repeating-layer cycle marked with `@@@@@@@@@@start`).
- Trace kernel id/name/grid ground truth: `/home/qshao/Project/Fun/gpu_traces/qwen14b/decode_traces/dynamic_trace.pb`,
  parsed via `/tmp/pb_py` (`trace_pb2`), kernel IDs 2600-2680.
- Simulator config: `SM90_H100_l2norm_l1dnorm` (`gpgpu_cache:dl2 ... A:192:8`, post MSHR-merge-cap fix,
  see `notes/decode-attn-qwen3-8b-14b-correlation.md` §4.6 for the fix history). Note: this config's L1C/L1T
  cache-invalidation bug (`notes/cache-invalidation-l1c-l1t-fix.md`) was fixed *after* the `flash_attn_ext_vec`
  number below was captured — that number is unaffected (the bug doesn't touch `global_space`/L1D accesses),
  but any future rerun of the other `_pending_` rows below should use a binary that includes both fixes.
- `flash_attn_ext_vec` simulator number (4.3154 μs) is from the verified embedded natural-layer-boundary
  run (`./log/tmp_log/embedded_mshr8.log`, kernel filter 2628-2641, 14B, post-MSHR-fix). All other
  simulator cells are pending a full-layer rerun (filter 2628-2649) — fill in once available.
