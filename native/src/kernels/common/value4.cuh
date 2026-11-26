#pragma once

#include "kernels/common/value4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace ingine::kernels::value4 {

struct Slice {
  std::uint32_t nibble;
  float scale;
};

template <typename Byte>
struct Head {
  Byte* packed;
  Byte* scale;
};

// Keep all eight packed heads contiguous and aligned to 16 bytes. The
// scales follow them, one per head. Each token occupies 544 bytes.
template <typename Byte>
__device__ __forceinline__ Head<Byte> head(Byte* cache, size_t index) {
  Byte* token = cache + (index / kHeads) * (kHeads * kHeadBytes);
  const int group = index % kHeads;
  return {token + group * kPackedBytes, token + kHeads * kPackedBytes + group * sizeof(float)};
}

__device__ __forceinline__ std::uint32_t encode(float x, float inverse) {
  // Shift the signed code by eight so sixteen nibbles cover [-8, 7]. The
  // symmetric quantizer uses [-7, 7], including a single zero representation.
  int q = __float2int_rn(x * inverse);
  return std::uint32_t(max(-7, min(7, q)) + 8);
}

__device__ __forceinline__ float decode(std::uint32_t nibble, float scale) {
  const float x = float(int(nibble & 15u) - 8) * scale;
  return __bfloat162float(__float2bfloat16_rn(x));
}

// One lane owns four consecutive channels; an entire warp owns one head.
__device__ __forceinline__ void store(Head<char> row, int lane, float (&values)[4]) {
  float maximum = 0.f;
#pragma unroll
  for (float x : values) maximum = fmaxf(maximum, fabsf(x));
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
  const float scale = maximum * (1.f / 7.f);
  const float inverse = scale > 0.f ? 1.f / scale : 0.f;
  std::uint16_t bits = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const std::uint32_t nibble = encode(values[i], inverse);
    bits |= std::uint16_t(nibble << (4 * i));
    values[i] = decode(nibble, scale);
  }
  reinterpret_cast<std::uint16_t*>(row.packed)[lane] = bits;
  if (lane == 0) *reinterpret_cast<float*>(row.scale) = scale;
}

__device__ __forceinline__ Slice load(Head<const char> row, int channel8) {
  return {__ldg(reinterpret_cast<const std::uint32_t*>(row.packed) + channel8),
          __ldg(reinterpret_cast<const float*>(row.scale))};
}

}  // namespace ingine::kernels::value4
