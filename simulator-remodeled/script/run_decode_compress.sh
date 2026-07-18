source ./setup_environment_no_git.sh release
OMP_NUM_THREADS=4 OMP_PROC_BIND=spread   ./bin/release/accel-sim.out \
-config ./gpgpu-sim/configs/tested-cfgs/SM90_H100/gpgpusim.config \
-config ./configs/tested-cfgs/SM90_H100/trace.config \
-is_extra_traces_enabled 1 -filter_first_kernel_id   2366 -filter_last_kernel_id 2367 \
-is_quantized_weight_dram_compression_enabled 1 -quantized_weight_compression_bits 3 -quantized_weight_region_file \
/home/qshao/Project/Fun/modern-gpu-simulator-micro-2025/simulator-remodeled/log/tmp_log/weight_regions.json -trace \
/home/qshao/Project/Fun/gpu_traces/qwen8B/modern/new_decode_traces/traces/dynamic_trace.pb
