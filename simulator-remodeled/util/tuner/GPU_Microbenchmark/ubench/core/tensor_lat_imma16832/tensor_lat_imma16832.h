#ifndef LAT_IMMA16832_DEF_H
#define LAT_IMMA16832_DEF_H

#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../../hw_def/hw_def.h"
#include "../imma16832_common/imma16832_mma.h"

#define REPEAT_ITERS 4096

#define IMMA_M 16
#define IMMA_N 8
#define IMMA_K 32
#define IMMA_OPERAND_BITS 8
#define IMMA_BIT_WORK (IMMA_M * IMMA_N * IMMA_K * IMMA_OPERAND_BITS)

__global__ void imma16832_latency(uint64_t *startClk, uint64_t *stopClk) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
  const int lane = threadIdx.x & 31;
  uint32_t A[4];
  uint32_t B[2];
  int32_t C[4];
  imma16832_init_regs(A, B, C, lane);

  asm volatile("bar.sync 0;");

  uint64_t start = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");

  for (int j = 0; j < REPEAT_ITERS; ++j) {
    imma16832_mma_inplace(C, A, B);
  }

  asm volatile("bar.sync 0;");

  uint64_t stop = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");

  if (lane == 0) {
    startClk[0] = start;
    stopClk[0] = stop;
  }

  if (C[0] == -1)
    startClk[0] = 0;
#else
  if (threadIdx.x == 0) {
    startClk[0] = 0;
    stopClk[0] = 0;
  }
#endif
}

inline float imma16832_lat() {
  intilizeDeviceProp(0);

  if (deviceProp.major < 8) {
    std::cerr << "IMMA m16n8k32 requires compute capability >= 8.0 (Ampere+)\n";
    return -1.f;
  }

  uint64_t *startClk = (uint64_t *)malloc(sizeof(uint64_t));
  uint64_t *stopClk = (uint64_t *)malloc(sizeof(uint64_t));
  uint64_t *startClk_g = nullptr;
  uint64_t *stopClk_g = nullptr;

  gpuErrchk(cudaMalloc(&startClk_g, sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, sizeof(uint64_t)));

  imma16832_latency<<<1, 32>>>(startClk_g, stopClk_g);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(startClk, startClk_g, sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopClk, stopClk_g, sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));

  const uint64_t total_time = stopClk[0] - startClk[0];
  const float cycles_per_imma =
      ((float)total_time) / ((float)REPEAT_ITERS);

  std::cout << "IMMA.16832.S8.S8 dependent-chain latency = " << cycles_per_imma
            << " (clk)\n";
  std::cout << "Total clock cycles = " << total_time << "\n";
  std::cout << "Verify SASS: make sass  (expect IMMA.16832.S8.S8)\n";

  cudaFree(startClk_g);
  cudaFree(stopClk_g);
  free(startClk);
  free(stopClk);

  return cycles_per_imma;
}

#endif
