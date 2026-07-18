# Build all representative decode kernel benchmarks.
#
#   make all    -> NATIVE sm_90 binaries (mmvq_bitwidth, mmq_bench, ...) for real hardware.
#                  DO NOT use these in GPGPU-Sim.
#   make exec   -> EXECUTION-DRIVEN binaries (*_exec: sm_70 + embedded PTX + --cudart shared)
#                  THIS is what the simulator / the SLURM sweep runs.
#
# On the cluster: `module load cuda` (match the sim's CUDA major version) BEFORE `make exec`,
# or you hit the "app must be compiled with same major version as the simulator" assertion.
# (mul_mat_vec_q omitted: its Makefile has no exec target; the sweep uses mmvq_bitwidth for the
#  software gemv and mul_mat_q for the software gemm.)
SUBDIRS = mmvq_bitwidth mul_mat_q rms_norm quantize_q8_1 flash_attn rope set_rows swiglu add

.PHONY: all exec run clean compile $(SUBDIRS)

compile:
	./compile_all.sh

# native (sm_90) build — real hardware, not the sim
all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

# execution-driven build for GPGPU-Sim — builds every *_exec, plus the NO_DEQUANT flash variant
exec:
	@for d in $(SUBDIRS); do echo "=== exec: $$d ==="; $(MAKE) -C $$d exec || exit 1; done
	@echo "=== exec: flash_attn NO_DEQUANT variant ==="
	nvcc -O3 -std=c++14 -DFUNCSIM_SAFE -DNO_DEQUANT \
	     -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70 \
	     --cudart shared -I common flash_attn/flash_attn_bench.cu -o flash_attn/flash_attn_nodequant_exec

clean:
	@for d in $(SUBDIRS); do $(MAKE) -C $$d clean; done
