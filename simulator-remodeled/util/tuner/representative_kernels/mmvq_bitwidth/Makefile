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

clean:
	rm -f mmvq_bitwidth mmvq_bitwidth_exec
