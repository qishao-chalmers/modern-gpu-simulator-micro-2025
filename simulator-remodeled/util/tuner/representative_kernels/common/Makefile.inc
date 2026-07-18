# Shared nvcc flags for representative decode kernel benches.
NVCC ?= nvcc
ARCH ?= -gencode=arch=compute_90,code=sm_90
NVCCFLAGS ?= -O3 -std=c++14 $(ARCH)
EXEC_ARCH = -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70
EXEC_FLAGS = -O3 -std=c++14 -DFUNCSIM_SAFE $(EXEC_ARCH) --cudart shared

COMMON_INC = -I../common
