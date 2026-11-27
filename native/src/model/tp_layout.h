#pragma once

#include <array>
#include <stdexcept>

namespace ingine::model {

inline constexpr int HEADS = 24, KV_HEADS = 8, HEAD_DIM = 128;
inline constexpr int HIDDEN = 2304, D_REL = 16, CONV_K = 4;
inline constexpr int INTER = 1152, VOCAB = 201024, UNPADDED_VOCAB = 200058;
inline constexpr int KV_DIM = KV_HEADS * HEAD_DIM;

enum Seg : int { SEG_Q = 0, SEG_GATE, SEG_REL, SEG_SINK, SEG_K, SEG_V, SEG_K_DYN, SEG_V_DYN, NUM_SEGS };
inline constexpr std::array<int, NUM_SEGS> SEG_ROWS = {
    HEADS * HEAD_DIM, HEADS * HEAD_DIM, HEADS * D_REL, HEADS, KV_DIM, KV_DIM, KV_HEADS * CONV_K, KV_HEADS * CONV_K};
inline constexpr std::array<int, NUM_SEGS> SEG_INPUT = {0, 0, 0, 0, 1, 2, 1, 2};
inline constexpr int num_segs(bool full) { return full ? NUM_SEGS : SEG_K_DYN; }

struct ExpertSlice {
  int begin;   // first logical intermediate channel
  int width;   // checkpoint channels
  int packed;  // channels including physical padding
};

struct TpLayout {
  const int tp_size, tp_rank;
  const int hkv_local, hq_local, kv_dim_local, q_dim_local;
  const int I_local, v_begin, v_end;

  constexpr explicit TpLayout(int size = 1, int rank = 0)
      : tp_size(checked_size(size)), tp_rank(checked_rank(rank, size)),
        hkv_local(KV_HEADS / size), hq_local(HEADS / size),
        kv_dim_local(hkv_local * HEAD_DIM), q_dim_local(hq_local * HEAD_DIM),
        I_local(INTER / size), v_begin(UNPADDED_VOCAB * rank / size),
        v_end(UNPADDED_VOCAB * (rank + 1) / size) {}

  constexpr int segment_rows(int segment) const { return SEG_ROWS[segment] / tp_size; }

  constexpr int weight_offset(int segment) const {
    int offset = 0;
    for (int s = 0; s < segment; ++s) offset += segment_rows(s);
    return offset;
  }

  // FP32 decode workspace. The sink is scalar-loaded, all other segments use 16-byte loads.
  constexpr int activation_offset(int segment) const {
    int offset = 0;
    for (int s = 0; s < segment; ++s) {
      offset += segment_rows(s);
      if (s + 1 != SEG_SINK) offset = align(offset, 4);
    }
    return offset;
  }

  constexpr int projection_rows(bool full) const { return weight_offset(num_segs(full)); }
  constexpr int projection_stride(bool full) const { return activation_offset(num_segs(full)); }

  constexpr ExpertSlice expert_slice(bool nvfp4) const {
    if (nvfp4) return {tp_rank * I_local, I_local, align(I_local, 32)};
    const int blocks = INTER / 32;
    const int base = blocks / tp_size, extra = blocks % tp_size;
    const int width = (base + (tp_rank < extra)) * 32;
    return {(tp_rank * base + (tp_rank < extra ? tp_rank : extra)) * 32, width, width};
  }

 private:
  static constexpr int align(int n, int a) { return (n + a - 1) / a * a; }
  static constexpr int checked_size(int size) {
    if (size != 1 && size != 2 && size != 4 && size != 8) throw std::invalid_argument("invalid TP size");
    return size;
  }
  static constexpr int checked_rank(int rank, int size) {
    if (rank < 0 || rank >= size) throw std::invalid_argument("invalid TP rank");
    return rank;
  }
};

static_assert(TpLayout(2).projection_stride(false) == 4300 && TpLayout(2).projection_stride(true) == 4332);

}  // namespace ingine::model
