OMP_NUM_THREADS=2 OMP_PROC_BIND=spread \
/home/qshao/Project/Fun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/bin/release/accel-sim.out -config \
/home/qshao/Project/Fun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_l2norm_l1dnorm/gpgpusim.config -config \
/home/qshao/Project/Fun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config \
 -is_extra_traces_enabled 1 -filter_first_kernel_id 2364 \
-filter_last_kernel_id 2400 \
-trace /home/qshao/Project/Fun/gpu_traces/qwen8B/modern/new_decode_traces/traces/dynamic_trace.pb \
| tee /home/qshao/Project/Fun/modern-gpu-simulator-micro-2025/simulator-remodeled/log/decode/sim.log
