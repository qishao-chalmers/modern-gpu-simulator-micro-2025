OMP_NUM_THREADS=8 OMP_PROC_BIND=spread ./gpu-simulator/bin/release/accel-sim.out \
-config ./gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_63GB/gpgpusim.config \
-config ./gpu-simulator/configs/tested-cfgs/SM90_H100_l2norm_l1dnorm/trace.config \
-is_extra_traces_enabled 1 \
-filter_first_kernel_id 2628 -filter_last_kernel_id 2650 \
-trace /home/qshao/Project/Fun/gpu_traces/qwen14b/decode_traces_ctrl/dynamic_trace.pb \
-is_quantized_weight_dram_compression_enabled 1 \
-quantized_weight_compression_bits 3 \
-quantized_weight_region_file /tmp/weight_regions_qwen14b.json
| tee /home/qshao/Project/Fun/modern-gpu-simulator-micro-2025/simulator-remodeled/log/decode/sim_decode_qwen14_3bit.log
