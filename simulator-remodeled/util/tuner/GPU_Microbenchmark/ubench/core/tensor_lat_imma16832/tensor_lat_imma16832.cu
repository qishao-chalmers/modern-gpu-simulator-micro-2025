#include "tensor_lat_imma16832.h"

#include <iostream>

int main() {
  intilizeDeviceProp(0);

  std::cout << "Device: " << deviceProp.name << " (sm_" << deviceProp.major
            << deviceProp.minor << ")\n";
  std::cout << "Benchmark: IMMA.16832.S8.S8 dependent-chain latency\n\n";

  if (deviceProp.major < 8) {
    std::cerr << "Need compute capability >= 8.0 (Ampere+) for m16n8k32 MMA\n";
    return 1;
  }

  const float lat = imma16832_lat();
  return lat < 0.f ? 1 : 0;
}
