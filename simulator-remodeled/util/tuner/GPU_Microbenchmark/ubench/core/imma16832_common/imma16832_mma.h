#ifndef IMMA16832_MMA_H
#define IMMA16832_MMA_H

#include <cuda_runtime.h>
#include <stdint.h>

// Matches llama.cpp mul_mat_q SASS: IMMA.16832.S8.S8
// PTX: mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)

__device__ __forceinline__ void imma16832_mma_inplace(int32_t *C, const uint32_t *A,
                                                    const uint32_t *B) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3},"
      "{%4, %5, %6, %7},"
      "{%8, %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
        "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
}

__device__ __forceinline__ void imma16832_init_regs(uint32_t *A, uint32_t *B,
                                                  int32_t *C, int lane) {
  A[0] = 0x01010101u + (uint32_t)lane;
  A[1] = 0x02020202u + (uint32_t)lane;
  A[2] = 0x03030303u + (uint32_t)lane;
  A[3] = 0x04040404u + (uint32_t)lane;
  B[0] = 0x11111111u + (uint32_t)lane;
  B[1] = 0x22222222u + (uint32_t)lane;
  C[0] = lane;
  C[1] = lane + 1;
  C[2] = lane + 2;
  C[3] = lane + 3;
}

#endif

#endif
