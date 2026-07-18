# Quantized-weight DRAM compression test

Models quantized-weight DRAM compression for GEMV/GEMM weight reads: for a read landing
in a configured weight-matrix address region, the DRAM-side request size is shrunk to
`bits/8` of the 8-bit baseline (data-transfer timing model only — `icnt`/L2/L1 and
functional correctness are unaffected, the response size is restored before leaving DRAM).

It is a **two-step** flow: first detect each kernel's weight-address region from the trace,
then run the simulator with that region file and the compression flags.

---

## Step 1 — generate the weight-region file

`script/gen_weight_regions.sh` wraps `script/detect_weight_regions.py`. It detects the
weight-matrix DRAM span of each GEMV/GEMM kernel (`mul_mat_vec_q` / `mul_mat_q`) purely
from the trace's own per-CTA recorded addresses; non-GEMM kernels are skipped. It also
(re)builds the protobuf modules into `/tmp/pb_py` automatically if missing.

```bash
cd simulator-remodeled
./script/gen_weight_regions.sh <dynamic_trace.pb> <kernels> [out.json]
```

### Inputs

| arg | required | description |
|-----|----------|-------------|
| `$1` TRACE   | **yes** | path to `dynamic_trace.pb`. Regions are detected from this trace's addresses, so it **must be the same trace you will simulate**. |
| `$2` KERNELS | **yes** | kernel id, range, or space-separated list: `2644`, `2622-2669`, or `"2644 2647 2668"`. Pass the full decode range to catch every weight kernel. |
| `$3` OUT     | no      | output JSON path. Default: `script/weight_regions.json`. |

### Example (qwen14b decode trace)

```bash
./script/gen_weight_regions.sh \
  /home/qshao/Project/Fun/gpu_traces/qwen14b/decode_traces/dynamic_trace.pb \
  2622-2669 \
  /tmp/weight_regions_qwen14b.json
```

Output is a `kernel_id -> {base, size, ...}` JSON. `/tmp` is ephemeral — pass an in-repo
path (e.g. `script/weight_regions_qwen14b.json`) if you want it to persist.

---

## Step 2 — run the simulator with compression

```bash
cd simulator-remodeled
source ./gpu-simulator/setup_environment_no_git.sh release
OMP_NUM_THREADS=8 OMP_PROC_BIND=spread ./gpu-simulator/bin/release/accel-sim.out \
  -config ./gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_63GB/gpgpusim.config \
  -config ./gpu-simulator/configs/tested-cfgs/SM90_H100_63GB/trace.config \
  -is_extra_traces_enabled 1 \
  -filter_first_kernel_id 2668 -filter_last_kernel_id 2668 \
  -is_quantized_weight_dram_compression_enabled 1 \
  -quantized_weight_compression_bits 3 \
  -quantized_weight_region_file /tmp/weight_regions_qwen14b.json \
  -trace /home/qshao/Project/Fun/gpu_traces/qwen14b/decode_traces/dynamic_trace.pb
```

### Compression flags

| flag | meaning |
|------|---------|
| `-is_quantized_weight_dram_compression_enabled 1` | turn the experiment on (default 0/off) |
| `-quantized_weight_compression_bits 3` | compressed bit-width over an 8-bit baseline (`3` = 3-bit → 3/8 of the DRAM transfer; use `2` for 2-bit, etc.; default 8) |
| `-quantized_weight_region_file <path>` | the JSON from Step 1 |

For the **baseline**, run the same command **without** the three `-...quantized...` flags and
diff `gpu_sim_cycle` / DRAM traffic.

---

## Notes

- Region file is **trace-specific** — regenerate per trace (the committed
  `script/weight_regions.json` is for the qwen8B trace, not qwen14b).
- Compression only affects **weight-read-bound** kernels (`mul_mat_vec_q` / `mul_mat_q`);
  on rms_norm / rope / quantize / flash_attn / k_set_rows it is a no-op (no region).
- qwen14b decode trace (2622–2669) weight kernels detected:
  `2622 2624 2627 2630 2634 2636 2644 2647 2649 2652 2656 2658 2665 2668`
  (largest: 2627 ≈ 826 MB; 2622/2647/2668 ≈ 284 MB).
- Reference invocation: `script/run_decode_compress.sh` (qwen8B, bits=3, kernels 2366–2367).
