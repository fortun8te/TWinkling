// Coupled exponential-race Gumbel noise shared between target sampler (head.cu) and DSpark drafter (markov.cu).
// Adapted from Siphon's race.cuh: token i at position pos scores logit_i / T - ln(E_i) where E_i ~ Exp(1) is a
// pure function of (seed, pos, i). Because target and draft use the exact same (seed, pos, i), coupled acceptance
// is exact equality (target[k] == draft[k]) with strictly higher acceptance than uncoupled rejection sampling.
#pragma once

#include <cstdint>

namespace ingine::kernels::race {

__device__ __forceinline__ unsigned long long mix64(unsigned long long z) {
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

__device__ __forceinline__ unsigned long long draw_key(unsigned long long seed, unsigned position) {
  return mix64(seed ^ mix64(static_cast<unsigned long long>(position) + 0x632be59bd9b4e019ull));
}

__device__ __forceinline__ float unit(unsigned long long z) {
  const float v = static_cast<float>(static_cast<unsigned>(z >> 32)) * 0x1p-32f +
                  static_cast<float>(static_cast<unsigned>(z)) * 0x1p-64f;
  return fmaxf(v, 0x1p-65f);
}

// -ln(E_i) for E_i ~ Exp(1) with full 24-bit tail precision.
__device__ __forceinline__ float boost(unsigned long long key, int token_id) {
  const unsigned long long z = mix64(key + (static_cast<unsigned long long>(token_id) + 1) * 0x9e3779b97f4a7c15ull);
  const float e = (z >> 63) == 0 ? -log1pf(-unit(z)) : -logf(unit(~z));
  return -logf(e);
}

}  // namespace ingine::kernels::race
