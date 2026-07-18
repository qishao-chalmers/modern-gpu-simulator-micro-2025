# Decode GEMV (mul_mat_vec_q) with real Q8_0 / Q4_K / Q2_K block layouts.
# Native:  make              -> mmvq_kquant
# Sim:     make exec         -> mmvq_kquant_exec (sm_70 + PTX, FUNCSIM_SAFE)
#
# 4-bit activations (y=q4_1, int4×int4 / dp8a):
#   make 4bit                -> mmvq_kquant_4bit
#   make 4bit_exec           -> mmvq_kquant_4bit_exec (software int4)
#   make 4bit_exec_dp8a      -> mmvq_kquant_4bit_exec_dp8a (PTX dp8a for Accel-Sim)
NVCC ?= nvcc
ARCH ?= -gencode=arch=compute_90,code=sm_90
NVCCFLAGS ?= -O3 -std=c++14 $(ARCH)

mmvq_kquant: mmvq_kquant.cu
	$(NVCC) $(NVCCFLAGS) -I.. mmvq_kquant.cu -o mmvq_kquant

exec: mmvq_kquant.cu
	$(NVCC) -O3 -std=c++14 -DFUNCSIM_SAFE \
	  -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70 \
	  --cudart shared -I.. mmvq_kquant.cu -o mmvq_kquant_exec

4bit: mmvq_kquant_4bit.cu
	$(NVCC) $(NVCCFLAGS) -I.. mmvq_kquant_4bit.cu -o mmvq_kquant_4bit

4bit_exec: mmvq_kquant_4bit.cu
	$(NVCC) -O3 -std=c++14 -DFUNCSIM_SAFE \
	  -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70 \
	  --cudart shared -I.. mmvq_kquant_4bit.cu -o mmvq_kquant_4bit_exec

# PTX-only fatbin so unknown opcode dp8a is not rejected by ptxas (no SASS).
4bit_exec_dp8a: mmvq_kquant_4bit.cu
	$(NVCC) -O3 -std=c++14 -DFUNCSIM_SAFE -DSIM_DP8A \
	  -gencode arch=compute_70,code=compute_70 \
	  --cudart shared -I.. mmvq_kquant_4bit.cu -o mmvq_kquant_4bit_exec_dp8a

clean:
	rm -f mmvq_kquant mmvq_kquant_exec \
	      mmvq_kquant_4bit mmvq_kquant_4bit_exec mmvq_kquant_4bit_exec_dp8a
