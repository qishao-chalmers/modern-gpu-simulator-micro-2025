```bash
# Prerequisites (Ubuntu/Debian)
sudo apt install protobuf-compiler libprotobuf-dev build-essential g++-10 cmake

# Set CUDA (adjust for your system)
export CUDA_INSTALL_PATH=/usr/local/cuda   # or /usr on Debian-packaged CUDA
export PATH=$CUDA_INSTALL_PATH/bin:$PATH

# Option A: Makefile build (default)
cd simulator-remodeled
./build.sh

# Option B: CMake build
BUILD_SYSTEM=cmake ./build.sh
# or manually:
cmake -S simulator-remodeled -B simulator-remodeled/build-cmake \
  -DCMAKE_CXX_COMPILER=g++-10 -DCMAKE_C_COMPILER=gcc-10
cmake --build simulator-remodeled/build-cmake -j

# Run bundled Rodinia Ampere example
./run_example.sh
```

Manual build steps:

```bash
cd simulator-remodeled

# Tracer (for generating traces on real GPU)
export CXX=g++-10          # required with CUDA 11.x
export ARCH=sm_80          # A100; use sm_90 for H100/Hopper (CUDA 12+)
./util/tracer_nvbit/install_nvbit.sh
make -C ./util/tracer_nvbit -j

# Simulator
source ./gpu-simulator/setup_environment_no_git.sh release
make -C ./gpu-simulator clean && make -j -C ./gpu-simulator/

# Simulate a protobuf trace
source ./gpu-simulator/setup_environment_no_git.sh release
OMP_NUM_THREADS=32 OMP_PROC_BIND=spread ./gpu-simulator/bin/release/accel-sim.out \
    -config ./gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTXA6000/gpgpusim.config \
    -config ./gpu-simulator/configs/tested-cfgs/SM86_RTXA6000/trace.config \
    -trace ./exampleTraces/extracted/rodinia2/12.8/backprop-rodinia-2.0-ft/4096___data_result_4096_txt/traces/dynamic_trace.pb
```

Example traces are in `exampleTraces/` (extract `rodinia2Ampere.tar.gz` for Ampere, etc.).

**Important:** Always `source gpu-simulator/setup_environment_no_git.sh` in the same shell before running `make` or `accel-sim.out`.
