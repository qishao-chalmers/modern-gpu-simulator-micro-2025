include ../common/Makefile.inc

EXE = rms_norm_bench
SRC = rms_norm_bench.cu

$(EXE): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(COMMON_INC) $(SRC) -o $(EXE)

exec: $(SRC)
	$(NVCC) $(EXEC_FLAGS) $(COMMON_INC) $(SRC) -o $(EXE)_exec

run: $(EXE)
	./$(EXE)

clean:
	rm -f $(EXE) $(EXE)_exec
