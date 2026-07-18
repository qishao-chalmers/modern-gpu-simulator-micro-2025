OMP_NUM_THREADS=16 OMP_PROC_BIND=spread ./gpu-simulator/bin/release/accel-sim.out \
-config ./gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_l2norm_l1dnorm/gpgpusim.config \
-config ./gpu-simulator/configs/tested-cfgs/SM90_H100_l2norm_l1dnorm/trace.config \
-is_extra_traces_enabled 1 -filter_first_kernel_id 2633 -filter_last_kernel_id 2639 \
-trace /home/qshao/Project/Fun/gpu_traces/qwen14b/decode_traces/dynamic_trace.pb | tee ./log/tmp_log/rope_2633_2639_scoreboard.log
#-debug_isolate_sm_id 0 \
#-subcore_issue_debug 1  -mem_request_trace_debug 1 -mem_request_trace_sm_id 0 \
