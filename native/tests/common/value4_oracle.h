#pragma once

#include "kernels/common/value4.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ingine::test {

template <typename Float, typename Round>
std::vector<std::uint8_t> pack_value4_oracle(std::vector<Float>& v, Round round) {
  using namespace kernels::value4;
  const std::size_t tokens = v.size() / (kHeads * kValues);
  std::vector<std::uint8_t> packed(tokens * kHeads * kHeadBytes);
  for (std::size_t t = 0; t < tokens; ++t) {
    auto* row = packed.data() + t * kHeads * kHeadBytes;
    for (int h = 0; h < kHeads; ++h) {
      auto* values = v.data() + (t * kHeads + h) * kValues;
      float peak = 0;
      for (int d = 0; d < kValues; ++d) peak = std::max(peak, std::abs(float(values[d])));
      const float scale = peak * (1.f / 7.f);
      const float inverse = scale > 0.f ? 1.f / scale : 0.f;
      std::memcpy(row + kHeads * kPackedBytes + h * sizeof(float), &scale, sizeof(scale));
      for (int d = 0; d < kValues; ++d) {
        const int code = std::clamp(int(std::nearbyint(float(values[d]) * inverse)), -7, 7);
        row[h * kPackedBytes + d / 2] |= std::uint8_t((code + 8) << ((d & 1) * 4));
        values[d] = round(float(code) * scale);
      }
    }
  }
  return packed;
}

template <typename Round>
std::vector<float> unpack_value4_oracle(const std::vector<std::uint8_t>& packed, Round round) {
  using namespace kernels::value4;
  const std::size_t tokens = packed.size() / (kHeads * kHeadBytes);
  std::vector<float> result(tokens * kHeads * kValues);
  for (std::size_t t = 0; t < tokens; ++t) {
    const auto* row = packed.data() + t * kHeads * kHeadBytes;
    for (int h = 0; h < kHeads; ++h) {
      float scale;
      std::memcpy(&scale, row + kHeads * kPackedBytes + h * sizeof(float), sizeof(scale));
      for (int d = 0; d < kValues; ++d) {
        const int code = ((row[h * kPackedBytes + d / 2] >> ((d & 1) * 4)) & 15) - 8;
        result[(t * kHeads + h) * kValues + d] = round(float(code) * scale);
      }
    }
  }
  return result;
}

}  // namespace ingine::test
