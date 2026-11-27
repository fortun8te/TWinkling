#include "kernels/draft/dflash2_selector.h"

namespace ingine::kernels::draft {
namespace {

constexpr int kSlots = 7;
constexpr int kCandidates = 16;
constexpr int kRank = 256;

__global__ void score_edges(const std::int32_t* ids, const float* unary,
                            const __nv_bfloat16* projected, const __nv_bfloat16* predecessors,
                            const __nv_bfloat16* successors, const std::int32_t* anchor,
                            float* lattice) {
  const int edge = blockIdx.x;
  const int previous = blockIdx.y;
  const int next = blockIdx.z;
  const int lane = threadIdx.x;
  const int predecessor = edge == 0 ? *anchor : ids[(edge - 1) * kCandidates + previous];
  const int candidate = ids[edge * kCandidates + next];
  float dot = 0.0f;
  for (int r = lane; r < kRank; r += 32) {
    const auto weighted = __float2bfloat16_rn(
        __bfloat162float(predecessors[predecessor * kRank + r]) *
        __bfloat162float(projected[edge * kRank + r]));
    dot = fmaf(__bfloat162float(weighted), __bfloat162float(successors[candidate * kRank + r]), dot);
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    dot += __shfl_down_sync(0xffffffffu, dot, offset);
  }
  if (lane == 0) {
    const float score = unary[edge * kCandidates + next] + __bfloat162float(__float2bfloat16_rn(dot));
    lattice[(edge * kCandidates + previous) * kCandidates + next] = 30.0f * tanhf(score / 30.0f);
  }
}

__global__ void walk(const std::int32_t* ids, const float* lattice, std::int32_t* tokens, float* confidence) {
  int previous = 0;
  for (int edge = 0; edge < kSlots; ++edge) {
    const float* scores = lattice + (edge * kCandidates + previous) * kCandidates;
    int best = 0;
    for (int next = 1; next < kCandidates; ++next) {
      if (scores[next] > scores[best]) best = next;
    }
    tokens[edge] = ids[edge * kCandidates + best];
    if (confidence) {
      float mass = 0;
      for (int next = 0; next < kCandidates; ++next) mass += expf(scores[next] - scores[best]);
      confidence[edge] = 1.f / mass;
    }
    previous = best;
  }
}

}  // namespace

void dflash2_greedy_selector(const std::int32_t* ids, const float* unary,
                            const __nv_bfloat16* projected, const __nv_bfloat16* predecessors,
                            const __nv_bfloat16* successors, const std::int32_t* anchor,
                            float* lattice, std::int32_t* tokens, cudaStream_t stream, float* confidence) {
  score_edges<<<dim3(kSlots, kCandidates, kCandidates), 32, 0, stream>>>(
      ids, unary, projected, predecessors, successors, anchor, lattice);
  walk<<<1, 1, 0, stream>>>(ids, lattice, tokens, confidence);
}

}  // namespace ingine::kernels::draft
