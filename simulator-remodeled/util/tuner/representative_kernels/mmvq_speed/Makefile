# SPEED Q||R strip GEMV (modes 1 / 2 CUDA; mode 3 = Q8 stand-in)
# Native:  make       -> mmvq_speed
# Sim:     make exec  -> mmvq_speed_exec (FUNCSIM_SAFE)
NVCC ?= nvcc
ARCH ?= -gencode=arch=compute_90,code=sm_90
NVCCFLAGS ?= -O3 -std=c++14 $(ARCH)

mmvq_speed: mmvq_speed.cu
	$(NVCC) $(NVCCFLAGS) -I.. mmvq_speed.cu -o mmvq_speed

exec: mmvq_speed.cu
	$(NVCC) -O3 -std=c++14 -DFUNCSIM_SAFE \
	  -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70 \
	  --cudart shared -I.. mmvq_speed.cu -o mmvq_speed_exec

# Shared-init error + speed: Q8 (=kquant q8_0) vs SPEED mode1/mode2
test_compare: test_compare.cu mmvq_speed.cu
	$(NVCC) $(NVCCFLAGS) -I.. test_compare.cu -o test_compare

clean:
	rm -f mmvq_speed mmvq_speed_exec test_compare
