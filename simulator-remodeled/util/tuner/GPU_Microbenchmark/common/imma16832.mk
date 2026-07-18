# IMMA.16832.S8.S8 benches: native sm_90 SASS on H100 (override common.mk multi-arch release).
GENCODE_SM30 :=
GENCODE_SM35 :=
GENCODE_SM50 :=
GENCODE_SM60 :=
GENCODE_SM61 :=
GENCODE_SM62 :=
GENCODE_SM70 :=
GENCODE_SM72 :=
GENCODE_SM75 :=
GENCODE_SM80 :=
GENCODE_SM86 :=
GENCODE_SM89 :=
GENCODE_SM90 :=

# Override for compile-only test on Ampere: GENCODE_ARCH='-gencode=arch=compute_80,code=sm_80' make release
GENCODE_ARCH ?= -gencode=arch=compute_90,code=sm_90
CUOPTS := $(GENCODE_ARCH)
NVCC_FLGAS += -std=c++14

BIN_DIR := $(abspath ../../../bin)

release:
	mkdir -p $(BIN_DIR)
	$(CC) $(NVCC_FLGAS) $(CUOPTS) $(SRC) -o $(EXE) -I$(CUDA_INC) -L$(LIB) -lcudart
	cp $(EXE) $(BIN_DIR)

sass:
	cuobjdump -sass ./$(EXE) | tee sass.txt
