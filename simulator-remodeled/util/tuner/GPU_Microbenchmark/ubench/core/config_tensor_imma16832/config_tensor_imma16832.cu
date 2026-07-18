// Profile IMMA.16832.S8.S8 on Hopper (H100) for remodeled Accel-Sim tensor knobs.

#include "../tensor_bw_imma16832/tensor_bw_imma16832.h"
#include "../tensor_lat_imma16832/tensor_lat_imma16832.h"

#include <cmath>
#include <iostream>

int main() {
  intilizeDeviceProp(0);

  std::cout << "Device: " << deviceProp.name << " (sm_" << deviceProp.major
            << deviceProp.minor << ")\n";
  std::cout << "Target SASS: IMMA.16832.S8.S8 (llama.cpp mul_mat_q)\n\n";

  if (deviceProp.major < 8) {
    std::cerr << "Need compute capability >= 8.0 (Ampere+) for m16n8k32 MMA\n";
    return 1;
  }

  const float latency_cycles = imma16832_lat();
  if (latency_cycles < 0) {
    return 1;
  }

  std::cout << "\n";
  const float throughput = imma16832_max_throughput();
  if (throughput < 0) {
    return 1;
  }

  if (!ACCEL_SIM_MODE) {
    return 0;
  }

  const unsigned cycles = std::max(1u, (unsigned)std::lround(latency_cycles));
  const unsigned initiation = std::max(1u, cycles / 2u);
  const unsigned latency = std::max(1u, cycles - initiation);
  const unsigned tensor_rate = round_up_2n((unsigned)std::lround(
      (float)IMMA_BIT_WORK / (float)cycles));
  const unsigned tensor_pipeline =
      std::max(32u, (unsigned)std::ceil(latency_cycles));

  std::cout << "\n// Accel-Sim remodeled SM config (from IMMA.16832.S8.S8):\n";
  std::cout << "// cycles = m*n*k*operand_bits / tensor_rate_per_cycle\n";
  std::cout << "// " << cycles << " = " << IMMA_BIT_WORK << " / "
            << tensor_rate << "\n";
  std::cout << "-tensor_rate_per_cycle " << tensor_rate << std::endl;
  std::cout << "-tensor_latency " << tensor_pipeline << std::endl;

  std::cout << "\n// Derived per-instruction model (generate_tensor_core_latencies):\n";
  std::cout << "// initiation_interval=" << initiation
            << " latency=" << latency << " (measured " << latency_cycles
            << " clk/IMMA)\n";

  std::cout << "\n// Classic trace/PTX path (reference only; not used by remodel SM):\n";
  std::cout << "-ptx_opcode_latency_tesnor " << cycles << std::endl;
  const float throughput_per_sched = throughput / (float)WARP_SCHEDS_PER_SM;
  const unsigned ptx_init =
      std::max(1u, (unsigned)std::lround((float)WARP_SIZE / throughput_per_sched));
  std::cout << "-ptx_opcode_initiation_tensor " << ptx_init << std::endl;
  std::cout << "-trace_opcode_latency_initiation_tensor "
            << (cycles / SASS_imma_per_PTX_mma) << ","
            << (ptx_init / SASS_imma_per_PTX_mma) << std::endl;

  return 0;
}
