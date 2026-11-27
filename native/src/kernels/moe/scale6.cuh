// Lossless 6-bit packing for NVFP4 E4M3 block scales (tjm8874/LeanSonic encoding).
// Every 16 consecutive E4M3 bytes spanning <= 64 codes (delta = byte - base in 0..63) pack little-endian
// into three uint32_t words (96 bits). The packed tensor adds one trailing zeroed uint32_t word so
// unaligned 32-bit loads for __funnelshift_r never read past the allocation.
#pragma once

#include <cstddef>
#include <cstdint>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace ingine::kernels::moe {

#ifdef __CUDACC__
__host__ __device__ __forceinline__
#else
inline
#endif
void pack16_6bit(const std::uint8_t* src, std::uint32_t* dst, std::uint8_t base) {
  std::uint32_t d[16];
  for (int i = 0; i < 16; ++i) d[i] = std::uint32_t(src[i] - base) & 0x3fu;
  dst[0] = d[0] | (d[1] << 6) | (d[2] << 12) | (d[3] << 18) | (d[4] << 24) | (d[5] << 30);
  dst[1] = (d[5] >> 2) | (d[6] << 4) | (d[7] << 10) | (d[8] << 16) | (d[9] << 22) | (d[10] << 28);
  dst[2] = (d[10] >> 4) | (d[11] << 2) | (d[12] << 8) | (d[13] << 14) | (d[14] << 20) | (d[15] << 26);
}

#ifdef __CUDACC__
__host__ __device__ __forceinline__
#else
inline
#endif
void unpack16_6bit(const std::uint32_t* src, std::uint8_t* dst, std::uint8_t base) {
  const std::uint32_t w0 = src[0], w1 = src[1], w2 = src[2];
  dst[0] = std::uint8_t((w0 & 0x3fu) + base);
  dst[1] = std::uint8_t(((w0 >> 6) & 0x3fu) + base);
  dst[2] = std::uint8_t(((w0 >> 12) & 0x3fu) + base);
  dst[3] = std::uint8_t(((w0 >> 18) & 0x3fu) + base);
  dst[4] = std::uint8_t(((w0 >> 24) & 0x3fu) + base);
  dst[5] = std::uint8_t((((w0 >> 30) | (w1 << 2)) & 0x3fu) + base);
  dst[6] = std::uint8_t(((w1 >> 4) & 0x3fu) + base);
  dst[7] = std::uint8_t(((w1 >> 10) & 0x3fu) + base);
  dst[8] = std::uint8_t(((w1 >> 16) & 0x3fu) + base);
  dst[9] = std::uint8_t(((w1 >> 22) & 0x3fu) + base);
  dst[10] = std::uint8_t((((w1 >> 28) | (w2 << 4)) & 0x3fu) + base);
  dst[11] = std::uint8_t(((w2 >> 2) & 0x3fu) + base);
  dst[12] = std::uint8_t(((w2 >> 8) & 0x3fu) + base);
  dst[13] = std::uint8_t(((w2 >> 14) & 0x3fu) + base);
  dst[14] = std::uint8_t(((w2 >> 20) & 0x3fu) + base);
  dst[15] = std::uint8_t(((w2 >> 26) & 0x3fu) + base);
}

#ifdef __CUDACC__
__device__ __forceinline__ std::uint8_t unpack1_6bit(const std::uint32_t* ws32, std::uint64_t idx,
                                                     std::uint8_t base) {
  const std::uint64_t bit = idx * 6;
  const std::size_t widx = std::size_t(bit >> 5);
  const std::uint32_t sh = std::uint32_t(bit & 31u);
  const std::uint32_t w0 = __ldg(ws32 + widx);
  const std::uint32_t w1 = (sh > 26) ? __ldg(ws32 + widx + 1) : 0u;
  const std::uint32_t v = __funnelshift_r(w0, w1, sh);
  return std::uint8_t((v & 0x3fu) + base);
}

__device__ __forceinline__ std::uint32_t unpack4_6bit(const std::uint32_t* ws32, std::uint64_t idx,
                                                      std::uint8_t base) {
  const std::uint64_t bit = idx * 6;
  const std::size_t widx = std::size_t(bit >> 5);
  const std::uint32_t sh = std::uint32_t(bit & 31u);
  const std::uint32_t w0 = __ldg(ws32 + widx);
  const std::uint32_t w1 = (sh > 8) ? __ldg(ws32 + widx + 1) : 0u;
  const std::uint32_t v = __funnelshift_r(w0, w1, sh);
  return ((v & 0x3fu) + base) | ((((v >> 6) & 0x3fu) + base) << 8) |
         ((((v >> 12) & 0x3fu) + base) << 16) | ((((v >> 18) & 0x3fu) + base) << 24);
}

__device__ __forceinline__ std::uint16_t unpack2_6bit_from_words(std::uint32_t w0, std::uint32_t w1,
                                                                 std::uint32_t sh, std::uint8_t base) {
  const std::uint32_t v = __funnelshift_r(w0, w1, sh);
  return std::uint16_t(((v & 0x3fu) + base) | ((((v >> 6) & 0x3fu) + base) << 8));
}
#endif

}  // namespace ingine::kernels::moe
