OMP_NUM_THREADS=8 OMP_PROC_BIND=spread \
/home/qshao/Project/Fun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/bin/release/accel-sim.out -config \
/home/qshao/Project/Fun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_l2norm_l1dnorm/gpgpusim.config -config \
/home/qshao/Project/Fun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config \
 -is_extra_traces_enabled 1 -filter_first_kernel_id 2628 \
-filter_last_kernel_id 2649 \
-trace /home/qshao/Project/Fun/gpu_traces/qwen14b/decode_traces/dynamic_trace.pb \
| tee /home/qshao/Project/Fun/modern-gpu-simulator-micro-2025/simulator-remodeled/log/decode/sim_decode_qwen14_2628_full_run_nobypss.log
