// Rebuild traces/dynamic_trace.pb from partial NVBit output (stats.csv + threadblocks/).
// Use only when the app did not exit cleanly and nvbit_at_ctx_term never wrote the file.
// Simulation still needs extra_info/enhanced_execution_info.json from a full tracer run.
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "../../traces_enhanced/pb_trace/include/trace.pb.h"
#include "../../traces_enhanced/pb_trace/include/gpu_device.pb.h"
#include "../../traces_enhanced/pb_trace/include/cuda_stream.pb.h"
#include "../../traces_enhanced/pb_trace/include/kernel.pb.h"
#include "../../traces_enhanced/pb_trace/include/threadblock.pb.h"
#include "../../traces_enhanced/pb_trace/include/dim3d.pb.h"

namespace fs = std::filesystem;

struct KernelStatsRow {
  int device_id = 0;
  int stream_id = 0;
  int kernel_id = 0;
  std::string kernel_name;
  unsigned grid_x = 1;
  unsigned grid_y = 1;
  unsigned grid_z = 1;
  unsigned block_x = 1;
  unsigned block_y = 1;
  unsigned block_z = 1;
  int shmem = 0;
  int nregs = 0;
};

static std::vector<std::string> split_csv_fields(const std::string &line) {
  std::vector<std::string> fields;
  std::stringstream ss(line);
  std::string item;
  while (std::getline(ss, item, ',')) {
    size_t start = item.find_first_not_of(" \t");
    size_t end = item.find_last_not_of(" \t");
    if (start == std::string::npos) {
      fields.push_back("");
    } else {
      fields.push_back(item.substr(start, end - start + 1));
    }
  }
  return fields;
}

static size_t count_pb_files(const fs::path &kernel_folder) {
  size_t count = 0;
  if (!fs::exists(kernel_folder)) {
    return count;
  }
  for (const auto &entry : fs::directory_iterator(kernel_folder)) {
    if (entry.is_regular_file() && entry.path().extension() == ".pb") {
      count++;
    }
  }
  return count;
}

static bool parse_stats_csv(const fs::path &stats_path,
                            std::map<int, KernelStatsRow> &rows) {
  std::ifstream in(stats_path);
  if (!in) {
    return false;
  }
  std::string line;
  std::getline(in, line);
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    auto fields = split_csv_fields(line);
    if (fields.size() < 12) {
      continue;
    }
    KernelStatsRow row;
    row.device_id = std::stoi(fields[0]);
    row.stream_id = std::stoi(fields[1]);
    std::smatch m;
    if (!std::regex_search(fields[2], m, std::regex(R"(kernel-(\d+)\.trace)"))) {
      continue;
    }
    row.kernel_id = std::stoi(m[1].str());
    row.kernel_name = fields[3];
    row.grid_x = std::stoul(fields[4]);
    row.grid_y = std::stoul(fields[5]);
    row.grid_z = std::stoul(fields[6]);
    row.block_x = std::stoul(fields[8]);
    row.block_y = std::stoul(fields[9]);
    row.block_z = std::stoul(fields[10]);
    rows[row.kernel_id] = row;
  }
  return true;
}

static std::vector<int> discover_kernel_ids(const fs::path &threadblocks_path) {
  std::vector<int> kernel_ids;
  std::regex kernel_dir(R"(kernel_(\d+))");
  if (!fs::exists(threadblocks_path)) {
    return kernel_ids;
  }
  for (const auto &device_entry : fs::directory_iterator(threadblocks_path)) {
    if (!device_entry.is_directory()) {
      continue;
    }
    for (const auto &stream_entry :
         fs::directory_iterator(device_entry.path())) {
      if (!stream_entry.is_directory()) {
        continue;
      }
      for (const auto &kernel_entry :
           fs::directory_iterator(stream_entry.path())) {
        if (!kernel_entry.is_directory()) {
          continue;
        }
        std::smatch m;
        const std::string name = kernel_entry.path().filename().string();
        if (std::regex_match(name, m, kernel_dir)) {
          kernel_ids.push_back(std::stoi(m[1].str()));
        }
      }
    }
  }
  std::sort(kernel_ids.begin(), kernel_ids.end());
  kernel_ids.erase(std::unique(kernel_ids.begin(), kernel_ids.end()),
                   kernel_ids.end());
  return kernel_ids;
}

static int read_function_unique_id(const fs::path &kernel_folder) {
  for (const auto &entry : fs::directory_iterator(kernel_folder)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".pb") {
      continue;
    }
    std::ifstream input(entry.path(), std::ios::binary);
    dynamic_trace::threadblock tb;
    if (!tb.ParseFromIstream(&input)) {
      continue;
    }
    for (const auto &warp_entry : tb.warps()) {
      const dynamic_trace::warp &wp = warp_entry.second;
      if (wp.instructions_size() == 0) {
        continue;
      }
      return wp.instructions(0).function_unique_id();
    }
  }
  return 0;
}

static fs::path kernel_folder_for(const fs::path &threadblocks_path, int kernel_id) {
  return threadblocks_path / "device_0" / "stream_0" /
         ("kernel_" + std::to_string(kernel_id));
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " /path/to/traces [binary_version]\n"
              << "  Rebuild dynamic_trace.pb from stats.csv and threadblocks/.\n"
              << "  Does NOT create enhanced_execution_info.json.\n";
    return 1;
  }

  const fs::path traces_dir = argv[1];
  const int binary_version = (argc >= 3) ? std::stoi(argv[2]) : 90;
  const fs::path stats_path = traces_dir / "stats.csv";
  const fs::path threadblocks_path = traces_dir / "threadblocks";
  const fs::path output_path = traces_dir / "dynamic_trace.pb";

  std::map<int, KernelStatsRow> stats_rows;
  if (fs::exists(stats_path)) {
    parse_stats_csv(stats_path, stats_rows);
  }

  const std::vector<int> kernel_ids = discover_kernel_ids(threadblocks_path);
  if (kernel_ids.empty()) {
    std::cerr << "ERROR: no threadblocks found under " << threadblocks_path
              << "\n";
    return 1;
  }

  dynamic_trace::Trace trace;
  trace.set_name("assembled_from_partial_traces");
  trace.set_binary_version(binary_version);
  trace.set_nvbit_version("assembled");
  trace.set_accelsim_version(4);
  trace.set_is_gathered_registers_values(false);

  dynamic_trace::gpu_device &gpu_dev = (*trace.mutable_gpu_device())[0];
  gpu_dev.set_id(0);
  dynamic_trace::cuda_stream &stream = (*gpu_dev.mutable_streams())[0];
  stream.set_id(0);

  for (int kernel_id : kernel_ids) {
    KernelStatsRow row;
    auto it = stats_rows.find(kernel_id);
    if (it != stats_rows.end()) {
      row = it->second;
    } else {
      row.kernel_id = kernel_id;
      row.kernel_name = "kernel_" + std::to_string(kernel_id) + "___0";
    }

    const fs::path kernel_folder = kernel_folder_for(threadblocks_path, kernel_id);
    const int function_unique_id = read_function_unique_id(kernel_folder);

    stream.add_ordered_cuda_events("kernel-" + std::to_string(kernel_id) + ".trace");
    dynamic_trace::kernel *ker = stream.add_kernels();
    ker->set_id(kernel_id);
    ker->set_name(row.kernel_name);
    ker->set_function_unique_id(function_unique_id);
    ker->set_size_shared_memory(row.shmem);
    ker->set_number_of_registers(row.nregs);
    ker->set_shared_memory_base_address(0);
    ker->set_local_memory_base_address(0);
    ker->mutable_grid_dim()->set_x(row.grid_x);
    ker->mutable_grid_dim()->set_y(row.grid_y);
    ker->mutable_grid_dim()->set_z(row.grid_z);
    ker->mutable_block_dim()->set_x(row.block_x);
    ker->mutable_block_dim()->set_y(row.block_y);
    ker->mutable_block_dim()->set_z(row.block_z);

    std::cout << "kernel " << kernel_id << ": name=" << row.kernel_name
              << " func_id=" << function_unique_id
              << " tbs=" << count_pb_files(kernel_folder) << "\n";
  }

  std::ofstream out(output_path, std::ios::binary);
  if (!out || !trace.SerializeToOstream(&out)) {
    std::cerr << "ERROR: failed to write " << output_path << "\n";
    return 1;
  }

  std::cout << "Wrote " << output_path << " (" << kernel_ids.size()
            << " kernel(s))\n";
  std::cout << "NOTE: run simulation only after you also have "
               "extra_info/enhanced_execution_info.json from a full tracer "
               "exit.\n";
  return 0;
}
