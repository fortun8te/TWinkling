// Replacement for thread-block clusters on sm_86 (INGINE_AMPERE): the CTAs of a "cluster" exchange data
// through global memory (L2) and meet at a software barrier.
//
// GroupBarrier is a counter and a generation in global memory, zero-initialised once (a __device__
// variable) and reusable across launches and CUDA graph replays: the last CTA to arrive resets the counter,
// then bumps the generation, which releases the others. Every CTA of the group must eventually arrive, so
// a kernel may not return between its first and last barrier, and all CTAs of the group must be able to be
// resident at once: the users (router, message_gates) launch at most 48 CTAs of one block per SM, far below
// the 82 SMs of an RTX 3090, so a waiting CTA never keeps another one of its group off the GPU. Barriers
// are per kernel and assume one launch at a time (ingine runs the decode chain on one stream).
//
// Data written before group_sync is visible after it to every CTA of the group when read through L2
// (__ldcg or volatile loads): L1 is not coherent between SMs.
#pragma once

#include <cstdint>

namespace ingine::kernels {

struct GroupBarrier {
  unsigned int count;
  unsigned int gen;
};

// All threads of the CTA call this. `wait` = false arrives without waiting (a CTA that exits right after).
__device__ __forceinline__ void group_sync(GroupBarrier* b, unsigned int n, bool wait = true) {
  __threadfence();  // this thread's global writes before the arrival
  __syncthreads();
  if (threadIdx.x == 0) {
    volatile unsigned int* gen = &b->gen;
    const unsigned int g0 = *gen;  // read before arriving: it cannot advance until this CTA arrived
    __threadfence();
    if (atomicAdd(&b->count, 1u) == n - 1) {
      atomicExch(&b->count, 0u);
      __threadfence();
      atomicAdd(&b->gen, 1u);
    } else if (wait) {
      while (*gen == g0) __nanosleep(32);
    }
    __threadfence();
  }
  __syncthreads();
}

}  // namespace ingine::kernels
