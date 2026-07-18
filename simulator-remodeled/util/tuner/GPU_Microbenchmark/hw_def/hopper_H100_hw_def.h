// H100 SXM5 (Hopper sm_90) — public architecture parameters for GPU_Microbenchmark.
// Sources: NVIDIA H100 datasheet, Hopper architecture whitepaper.
#ifndef HOPPER_H100_HW_DEF_H
#define HOPPER_H100_HW_DEF_H

#include "./common/common.h"
#include "./common/deviceQuery.h"

// 256 KiB configurable shared / L1 per SM (max shared mode)
#define L1_SIZE (256 * 1024)

// Match profiled prefill comparison clock unless overridden at runtime.
#define CLK_FREQUENCY 1620

#define ISSUE_MODEL issue_model::single
#define CORE_MODEL core_model::subcore
#define DRAM_MODEL dram_model::HBM
#define WARP_SCHEDS_PER_SM 4

// One CUDA wmma m16n8k32 s8 mma_sync maps to one IMMA.16832.S8.S8 per warp.
#define SASS_imma_per_PTX_mma 1

// Legacy name used by FP16 config_tensor (not used for IMMA16832 bench).
#define SASS_hmma_per_PTX_wmma 2

// H100: 50 MiB L2, 10 × 512-bit HBM3 controllers.
#define L2_BANKS_PER_MEM_CHANNEL 1
#define L2_BANK_WIDTH_in_BYTE 128

#endif
