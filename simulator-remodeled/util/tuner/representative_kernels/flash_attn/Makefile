include ../common/Makefile.inc

EXE = flash_attn_bench
SRC = flash_attn_bench.cu

$(EXE): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(COMMON_INC) $(SRC) -o $(EXE)

exec: $(SRC)
	$(NVCC) $(EXEC_FLAGS) $(COMMON_INC) $(SRC) -o $(EXE)_exec

run: $(EXE)
	./$(EXE)

clean:
	rm -f $(EXE) $(EXE)_exec
