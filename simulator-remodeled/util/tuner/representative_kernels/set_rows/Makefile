include ../common/Makefile.inc

EXE = set_rows_bench
SRC = set_rows_bench.cu

$(EXE): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(COMMON_INC) $(SRC) -o $(EXE)

exec: $(SRC)
	$(NVCC) $(EXEC_FLAGS) $(COMMON_INC) $(SRC) -o $(EXE)_exec

run: $(EXE)
	./$(EXE)

clean:
	rm -f $(EXE) $(EXE)_exec
