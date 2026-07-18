# Standalone mul_mat_vec_q (Q8_0 GEMV) benchmark.
# Default targets sm_90 (H100). Override for other archs / compile tests:
#   make ARCH='-gencode=arch=compute_80,code=sm_80'
NVCC ?= nvcc
ARCH ?= -gencode=arch=compute_90,code=sm_90
NVCCFLAGS ?= -O3 -std=c++14 $(ARCH)

EXE = mmvq_bench

$(EXE): mmvq_bench.cu shapes_qwen3_14b.h
	$(NVCC) $(NVCCFLAGS) mmvq_bench.cu -o $(EXE)

run: $(EXE)
	./$(EXE)

# emit the SASS (to confirm __dp4a / load pattern under the target arch)
sass: $(EXE)
	cuobjdump -sass ./$(EXE) | tee sass.txt

clean:
	rm -f $(EXE) sass.txt
