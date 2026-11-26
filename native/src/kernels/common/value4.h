#pragma once

namespace ingine::kernels::value4 {

// One global value head: 128 signed four-bit values and one float scale.
inline constexpr int kValues = 128, kHeads = 8;
inline constexpr int kPackedBytes = kValues / 2;
inline constexpr int kHeadBytes = kPackedBytes + sizeof(float);

}  // namespace ingine::kernels::value4
