#ifndef BW_IMMA16832_DEF_H
#define BW_IMMA16832_DEF_H

#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../../hw_def/hw_def.h"
#include "../imma16832_common/imma16832_mma.h"

#define REPEAT_TIMES 2048

__global__ void imma16832_max_throughput(uint64_t *startClk, uint64_t *stopClk) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
  const int lane = threadIdx.x & 31;
  uint32_t A[4];
  uint32_t B[2];
  int32_t C[4];
  imma16832_init_regs(A, B, C, lane);

  asm volatile("bar.sync 0;");

  uint64_t start = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");

  for (int j = 0; j < REPEAT_TIMES; ++j) {
    imma16832_mma_inplace(C, A, B);
  }

  asm volatile("bar.sync 0;");

  uint64_t stop = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");

  startClk[threadIdx.x] = start;
  stopClk[threadIdx.x] = stop;

  if (C[0] == -1)
    startClk[threadIdx.x] = 0;
#else
  startClk[threadIdx.x] = 0;
  stopClk[threadIdx.x] = 0;
#endif
}

inline float imma16832_max_throughput() {
  intilizeDeviceProp(0);

  if (deviceProp.major < 8) {
    std::cerr << "IMMA m16n8k32 requires compute capability >= 8.0 (Ampere+)\n";
    return -1.f;
  }

  THREADS_PER_BLOCK = deviceProp.maxThreadsPerBlock;
  BLOCKS_NUM = 1;
  TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_NUM;

  uint64_t *startClk =
      (uint64_t *)malloc(TOTAL_THREADS * sizeof(uint64_t));
  uint64_t *stopClk = (uint64_t *)malloc(TOTAL_THREADS * sizeof(uint64_t));
  uint64_t *startClk_g = nullptr;
  uint64_t *stopClk_g = nullptr;

  gpuErrchk(cudaMalloc(&startClk_g, TOTAL_THREADS * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, TOTAL_THREADS * sizeof(uint64_t)));

  imma16832_max_throughput<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(startClk_g,
                                                              stopClk_g);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(startClk, startClk_g,
                       TOTAL_THREADS * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));

  const uint64_t total_time =
      *std::max_element(&stopClk[0], &stopClk[TOTAL_THREADS]) -
      *std::min_element(&startClk[0], &startClk[TOTAL_THREADS]);
  const float warp_inst_bw =
      ((float)(REPEAT_TIMES * (TOTAL_THREADS / WARP_SIZE))) / (float)total_time;
  const float sass_issue_bw =
      ((float)(REPEAT_TIMES * (TOTAL_THREADS / WARP_SIZE) *
               SASS_imma_per_PTX_mma)) /
      (float)total_time;

  std::cout << "IMMA.16832 warp-MMA issue bandwidth = " << warp_inst_bw
            << " (inst/clk/SM)\n";
  std::cout << "IMMA.16832 SASS issue bandwidth = " << sass_issue_bw
            << " (inst/clk/SM)\n";
  std::cout << "Total clock cycles = " << total_time << "\n";

  cudaFree(startClk_g);
  cudaFree(stopClk_g);
  free(startClk);
  free(stopClk);

  return warp_inst_bw;
}

#endif
