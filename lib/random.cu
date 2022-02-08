#include <stdio.h>
#include <string.h>
#include <iostream>

#include <quda_internal.h>
#include <tune_quda.h>
#include <random_quda.h>
#include <comm_quda.h>
#include <index_helper.cuh>

#define BLOCKSDIVUP(a, b)  (((a)+(b)-1)/(b))

namespace quda {

  dim3 GetGridDim(size_t threads, size_t size) {
    int blockx = BLOCKSDIVUP(size, threads);
    dim3 blocks(blockx,1,1);
    return blocks;
  }

  struct rngArg {
    size_t commDim[QUDA_MAX_DIM];
    int commCoord[QUDA_MAX_DIM];
    int X[QUDA_MAX_DIM];
    rngArg(const int X_[]) {
      for (int i=0; i<4; i++) {
        commDim[i] = comm_dim(i);
        commCoord[i] = comm_coord(i);
        X[i] = X_[i];
      }
    }
  };

  /**
     @brief CUDA kernel to initialize CURAND RNG states
     @param state CURAND RNG state array
     @param seed initial seed for RNG
     @param size size of the CURAND RNG state array
     @param arg Metadata needed for computing multi-gpu offsets
  */
  __global__ void kernel_random(cuRNGState *state, unsigned long long seed, int size_cb, rngArg arg)
  {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int parity = blockIdx.y * blockDim.y + threadIdx.y;
    if (id < size_cb) {
      // Each thread gets same seed, a different sequence number, no offset
      int x[4];
      getCoords(x, id, arg.X, parity);
      for (int i = 0; i < 4; i++) x[i] += arg.commCoord[i] * arg.X[i];
      size_t idd
        = (((x[3] * arg.commDim[2] * arg.X[2] + x[2]) * arg.commDim[1] * arg.X[1]) + x[1]) * arg.commDim[0] * arg.X[0]
        + x[0];
      curand_init(seed, idd, 0, &state[parity * size_cb + id]);
    }
  }

  /**
     @brief Call CUDA kernel to initialize CURAND RNG states
     @param state CURAND RNG state array
     @param seed initial seed for RNG
     @param size_cb Checkerboarded size of the CURAND RNG state array
     @param n_parity Number of parities (1 or 2)
     @param X array of lattice dimensions
  */
  void launch_kernel_random(cuRNGState *state, unsigned long long seed, int size_cb, int n_parity, int X[4])
  {
    TuneParam tp;
    tp.block = dim3(128, 1, 1);
    tp.grid = GetGridDim(tp.block.x, size_cb);
    rngArg arg(X);
    tp.block.y = n_parity;
    qudaLaunchKernel(kernel_random, tp, 0, state, seed, size_cb, arg);
    qudaDeviceSynchronize();
  }

  RNG::RNG(const LatticeField &meta, unsigned long long seedin) :
    seed(seedin),
    size(meta.Volume()),
    size_cb(meta.VolumeCB())
  {
    state = nullptr;
    for (int i = 0; i < 4; i++) X[i] = meta.X()[i];
#if defined(XORWOW)
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Using curandStateXORWOW\n");
#elif defined(RG32k3a)
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Using curandStateMRG32k3a\n");
#else
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Using curandStateMRG32k3a\n");
#endif
  }

  RNG::RNG(const LatticeFieldParam &param, unsigned long long seedin) : seed(seedin), size(1), size_cb(1)
  {
    state = nullptr;
    for (int i = 0; i < 4; i++) {
      X[i] = param.x[i];
      size *= X[i];
    }
    size_cb = size / param.siteSubset;

#if defined(XORWOW)
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Using curandStateXORWOW\n");
#elif defined(RG32k3a)
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Using curandStateMRG32k3a\n");
#else
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Using curandStateMRG32k3a\n");
#endif
  }

  /**
     @brief Initialize CURAND RNG states
  */
  void RNG::Init() {
    AllocateRNG();
    launch_kernel_random(state, seed, size_cb, size / size_cb, X);
  }

  /**
     @brief Allocate Device memory for CURAND RNG states
  */
  void RNG::AllocateRNG() {
    if (size > 0 && state == nullptr) {
      state = (cuRNGState *)device_malloc(size * sizeof(cuRNGState));
      qudaMemset(state, 0, size * sizeof(cuRNGState));
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
        printfQuda("Allocated array of random numbers with size: %.2f MB\n",
                   size * sizeof(cuRNGState) / (float)(1048576));
    } else {
      errorQuda("Array of random numbers not allocated, array size: %d !\nExiting...\n", size);
    }
  }

  /**
     @brief Release Device memory for CURAND RNG states
  */
  void RNG::Release() {
    if (size > 0 && state != nullptr) {
      device_free(state);
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
        printfQuda("Free array of random numbers with size: %.2f MB\n", size * sizeof(cuRNGState) / (float)(1048576));
      size = 0;
      state = NULL;
    }
  }

  /*! @brief Backup CURAND array states initialization */
  void RNG::backup()
  {
    backup_state = (cuRNGState *)safe_malloc(size * sizeof(cuRNGState));
    qudaMemcpy(backup_state, state, size * sizeof(cuRNGState), cudaMemcpyDeviceToHost);
  }

  /*! @brief Restore CURAND array states initialization */
  void RNG::restore()
  {
    qudaMemcpy(state, backup_state, size * sizeof(cuRNGState), cudaMemcpyHostToDevice);
    host_free(backup_state);
  }

} // namespace quda
