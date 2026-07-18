# Qwen3 decoder-layer assembly (three drivers sharing layer_kernels.cuh).
#   all      = full layer (reference)
#   gemm     = weight GEMV/GEMM only  (re-run per QWC_BITS for the compression study)
#   nongemm  = everything else        (run once, reuse)
#
# Native sm_90 (trace target):   make            -> layer_{all,gemm,nongemm}_bench
# Exec-driven sm_70 (GPGPU-Sim): make exec        -> *_exec  (FUNCSIM_SAFE, dp4a/fp16 off)
NVCC ?= nvcc
ARCH ?= -gencode=arch=compute_90,code=sm_90
NVCCFLAGS ?= -O3 -std=c++14 $(ARCH)
EXEC_FLAGS = -O3 -std=c++14 -DFUNCSIM_SAFE \
  -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70 --cudart shared

DRIVERS = all gemm nongemm

native: $(addsuffix _native,$(DRIVERS))
exec:   $(addsuffix _exec,$(DRIVERS))
all: native

%_native: layer_%_bench.cu layer_kernels.cuh
	$(NVCC) $(NVCCFLAGS) $< -o layer_$*_bench
%_exec: layer_%_bench.cu layer_kernels.cuh
	$(NVCC) $(EXEC_FLAGS) $< -o layer_$*_bench_exec

clean:
	rm -f layer_all_bench layer_gemm_bench layer_nongemm_bench \
	      layer_all_bench_exec layer_gemm_bench_exec layer_nongemm_bench_exec

.PHONY: native exec all clean
