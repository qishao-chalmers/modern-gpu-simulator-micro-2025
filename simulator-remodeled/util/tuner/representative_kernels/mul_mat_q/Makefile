# mul_mat_q (Q8_0 batched GEMM) benchmark.
# Native run (sm_90):           make
# Execution-driven (PTX) build: make exec   (embeds sm_70 SASS + PTX for GPGPU-Sim)
NVCC ?= nvcc
ARCH ?= -gencode=arch=compute_90,code=sm_90
NVCCFLAGS ?= -O3 -std=c++14 $(ARCH)

mmq_bench: mmq_bench.cu
	$(NVCC) $(NVCCFLAGS) mmq_bench.cu -o mmq_bench

# build for GPGPU-Sim execution-driven mode (Volta PTX, both SASS+PTX embedded)
exec: mmq_bench.cu
	$(NVCC) -O3 -std=c++14 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70 \
	  --cudart shared mmq_bench.cu -o mmq_bench_exec

# emit SASS to confirm __dp4a + load pattern under the target arch
sass: mmq_bench
	cuobjdump -sass ./mmq_bench | tee sass.txt

clean:
	rm -f mmq_bench mmq_bench_exec sass.txt
