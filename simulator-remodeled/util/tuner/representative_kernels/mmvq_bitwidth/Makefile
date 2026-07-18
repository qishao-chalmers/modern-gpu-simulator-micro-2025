# mul_mat_vec_q weight-bit-width sweep.
# Native run (sm_90):           make
# Execution-driven (PTX) build: make exec     (embeds sm_70 SASS + PTX for GPGPU-Sim)
NVCC ?= nvcc
ARCH ?= -gencode=arch=compute_90,code=sm_90
NVCCFLAGS ?= -O3 -std=c++14 $(ARCH)

mmvq_bitwidth: mmvq_bitwidth.cu
	$(NVCC) $(NVCCFLAGS) mmvq_bitwidth.cu -o mmvq_bitwidth

# build for GPGPU-Sim execution-driven mode (Volta PTX, both SASS+PTX embedded)
exec: mmvq_bitwidth.cu
	$(NVCC) -O3 -std=c++14 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70 \
	  --cudart shared mmvq_bitwidth.cu -o mmvq_bitwidth_exec

# same, but REAL __dp4a int8 dot (no FUNCSIM_SAFE compute inflation) — for the bandwidth-bound
# QWC study. Needs the sim's functional dp4a support (present in this build).
exec_dp4a: mmvq_bitwidth.cu
	$(NVCC) -O3 -std=c++14 -DUSE_DP4A -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70 \
	  --cudart shared mmvq_bitwidth.cu -o mmvq_bitwidth_dp4a_exec

clean:
	rm -f mmvq_bitwidth mmvq_bitwidth_exec mmvq_bitwidth_dp4a_exec
