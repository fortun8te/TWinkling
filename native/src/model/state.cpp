// model.State and fast.Buffers allocation.
#include "model/runtime.h"

#include "core/device.h"
#include "kernels/common/value4.h"

#include <stdexcept>

namespace ingine::model {
namespace {

Tensor zeros(std::vector<DeviceBuffer>& storage, DType dt, std::initializer_list<std::int32_t> shape) {
  std::size_t n = 1;
  for (std::int32_t d : shape) n *= std::size_t(d);
  storage.emplace_back(n * dtype_size(dt));
  CUDA_CHECK(cudaMemset(storage.back().p, 0, storage.back().bytes));
  return rm(storage.back().p, dt, shape);
}

}  // namespace

Tensor State::k_ring_at(int ring) const {
  return rm(ptr<char>(k_ring) + std::size_t(ring) * WINDOW * KV_DIM * 2, DType::BF16, {WINDOW, KV_HEADS, HEAD_DIM});
}
Tensor State::v_ring_at(int ring) const {
  return rm(ptr<char>(v_ring) + std::size_t(ring) * WINDOW * KV_DIM * 2, DType::BF16, {WINDOW, KV_HEADS, HEAD_DIM});
}
std::int64_t* State::ring_pos_at(int ring) const { return ptr<std::int64_t>(ring_pos) + std::size_t(ring) * WINDOW; }
Tensor State::k_full_at(int full) const {
  if (kv_k4) {
    constexpr int bytes = kernels::value4::kHeadBytes;
    void* data = elastic() ? elastic_k.at(full).data() : ptr<char>(k_full) + std::size_t(full) * max_ctx * KV_HEADS * bytes;
    return rm(data, DType::U8, {max_ctx, KV_HEADS, bytes});
  }
  const std::size_t es = kv_fp8 ? 1 : 2;
  void* data = elastic() ? elastic_k.at(full).data() : ptr<char>(k_full) + std::size_t(full) * max_ctx * KV_DIM * es;
  return rm(data, kv_fp8 ? DType::FP8_E4M3FN : DType::BF16, {max_ctx, KV_HEADS, HEAD_DIM});
}
Tensor State::v_full_at(int full) const {
  if (kv_v4) {
    constexpr int bytes = kernels::value4::kHeadBytes;
    void* data = elastic() ? elastic_v.at(full).data() : ptr<char>(v_full) + std::size_t(full) * max_ctx * KV_HEADS * bytes;
    return rm(data, DType::U8, {max_ctx, KV_HEADS, bytes});
  }
  const std::size_t es = kv_fp8 ? 1 : 2;
  void* data = elastic() ? elastic_v.at(full).data() : ptr<char>(v_full) + std::size_t(full) * max_ctx * KV_DIM * es;
  return rm(data, kv_fp8 ? DType::FP8_E4M3FN : DType::BF16, {max_ctx, KV_HEADS, HEAD_DIM});
}

void State::ensure_full_rows(int rows) {
  if (!elastic()) return;
  if (rows < 0 || rows > max_ctx) throw std::invalid_argument("elastic KV rows exceed max_ctx");
  const std::size_t k_bytes = kv_k4 ? std::size_t(rows) * KV_HEADS * kernels::value4::kHeadBytes
                                    : std::size_t(rows) * KV_DIM * (kv_fp8 ? 1 : 2);
  const std::size_t v_bytes = kv_v4 ? std::size_t(rows) * KV_HEADS * kernels::value4::kHeadBytes
                                    : std::size_t(rows) * KV_DIM * (kv_fp8 ? 1 : 2);
  std::array<std::size_t, NUM_FULL_VISITS> old_k{}, old_v{};
  for (int i = 0; i < NUM_FULL_VISITS; ++i) {
    old_k[i] = elastic_k[i].mapped_bytes();
    old_v[i] = elastic_v[i].mapped_bytes();
  }
  try {
    for (int i = 0; i < NUM_FULL_VISITS; ++i) {
      elastic_k[i].grow(k_bytes);
      elastic_v[i].grow(v_bytes);
    }
  } catch (...) {
    for (int i = 0; i < NUM_FULL_VISITS; ++i) {
      elastic_k[i].rollback(old_k[i]);
      elastic_v[i].rollback(old_v[i]);
    }
    throw;
  }
}

State make_state(int max_ctx, bool kv_fp8, bool kv_v4, bool elastic_kv, bool kv_k4) {
  if (kv_v4 && !kv_fp8) throw std::invalid_argument("INT4 global values require FP8 global keys");
  if (kv_k4 && !kv_v4) throw std::invalid_argument("INT4 global keys require INT4 global values");
  // the attention kernel keeps positions in int32 and caps full caches at 2^30 rows
  if (max_ctx < 1 || max_ctx > (1 << 30)) throw std::invalid_argument("max_ctx must be in 1..2^30");
  State st;
  st.max_ctx = max_ctx;
  st.kv_fp8 = kv_fp8;
  st.kv_v4 = kv_v4;
  st.kv_k4 = kv_k4;
  const DType full = kv_fp8 ? DType::FP8_E4M3FN : DType::BF16;
  st.k_ring = zeros(st.storage, DType::BF16, {NUM_SLIDING_VISITS, WINDOW, KV_HEADS, HEAD_DIM});
  st.v_ring = zeros(st.storage, DType::BF16, {NUM_SLIDING_VISITS, WINDOW, KV_HEADS, HEAD_DIM});
  st.ring_pos = zeros(st.storage, DType::I64, {NUM_SLIDING_VISITS, WINDOW});
  CUDA_CHECK(cudaMemset(st.ring_pos.data, 0xff, st.storage.back().bytes));  // -1
  if (elastic_kv) {
    const std::size_t k_bytes = kv_k4 ? std::size_t(max_ctx) * KV_HEADS * kernels::value4::kHeadBytes
                                      : std::size_t(max_ctx) * KV_DIM * dtype_size(full);
    const std::size_t v_bytes = kv_v4 ? std::size_t(max_ctx) * KV_HEADS * kernels::value4::kHeadBytes
                                      : std::size_t(max_ctx) * KV_DIM * dtype_size(full);
    for (int i = 0; i < NUM_FULL_VISITS; ++i) {
      st.elastic_k[i] = GrowableBuffer(k_bytes);
      st.elastic_v[i] = GrowableBuffer(v_bytes);
    }
  } else {
    st.k_full = kv_k4 ? zeros(st.storage, DType::U8, {NUM_FULL_VISITS, max_ctx, KV_HEADS, kernels::value4::kHeadBytes})
                      : zeros(st.storage, full, {NUM_FULL_VISITS, max_ctx, KV_HEADS, HEAD_DIM});
    st.v_full = kv_v4 ? zeros(st.storage, DType::U8, {NUM_FULL_VISITS, max_ctx, KV_HEADS, kernels::value4::kHeadBytes})
                      : zeros(st.storage, full, {NUM_FULL_VISITS, max_ctx, KV_HEADS, HEAD_DIM});
  }
  st.conv_k = zeros(st.storage, DType::FP32, {NUM_VISITS, CONV_K - 1, KV_DIM});
  st.conv_v = zeros(st.storage, DType::FP32, {NUM_VISITS, CONV_K - 1, KV_DIM});
  st.conv_attn = zeros(st.storage, DType::FP32, {NUM_VISITS, CONV_K - 1, HIDDEN});
  st.conv_mlp = zeros(st.storage, DType::FP32, {NUM_VISITS, CONV_K - 1, HIDDEN});
  st.length = 0;
  return st;
}

void reset_state(State& st, cudaStream_t stream) {
  CUDA_CHECK(cudaMemsetAsync(st.ring_pos.data, 0xff, st.ring_pos.bytes(), stream));
  for (Tensor* t : {&st.conv_k, &st.conv_v, &st.conv_attn, &st.conv_mlp})
    CUDA_CHECK(cudaMemsetAsync(t->data, 0, t->bytes(), stream));
  st.length = 0;
}

Buffers make_buffers(int m) {
  if (m < 1 || m > 8) throw std::invalid_argument("call width must be in 1..8");
  Buffers b;
  b.m = m;
  auto& s = b.storage;
  const int H = HIDDEN;
  b.token = zeros(s, DType::I64, {m});
  b.B0 = zeros(s, DType::FP32, {m, H});
  b.B1 = zeros(s, DType::FP32, {m, H});
  for (int p = 0; p < PASSES; ++p) {
    b.B2[p] = zeros(s, DType::FP32, {m, H});
    b.E[p] = zeros(s, DType::FP32, {m, H});
  }
  b.hid = zeros(s, DType::FP32, {m, H});
  b.part = zeros(s, DType::FP32, {m, H});
  b.y = zeros(s, DType::FP32, {m, H});
  b.y2 = zeros(s, DType::FP32, {m, H});
  b.xin = zeros(s, DType::BF16, {3, m, H});
  b.mixf = zeros(s, DType::FP32, {3, m, H});
  b.msgs = zeros(s, DType::FP32, {4, m, H});
  b.xm = zeros(s, DType::BF16, {1, m, H});
  b.mixm = zeros(s, DType::FP32, {1, m, H});
  b.proj_sliding = zeros(s, DType::FP32, {m, PROJ_ROWS_SLIDING});
  b.proj_full = zeros(s, DType::FP32, {m, PROJ_ROWS_FULL});
  b.attn_out = zeros(s, DType::BF16, {m, HEADS * HEAD_DIM});
  b.idx = zeros(s, DType::I32, {m, TOP_K});
  b.wts = zeros(s, DType::FP32, {m, TOP_K});
  b.act = zeros(s, DType::BF16, {m, TOP_K, INTER});
  b.xf = zeros(s, DType::BF16, {1, m, H});
  b.logits = zeros(s, DType::FP32, {m, UNPADDED_VOCAB});
  b.taps = zeros(s, DType::BF16, {m, NUM_TAPS, H});
  b.hist_attn = zeros(s, DType::FP32, {NUM_VISITS, m, H});
  b.hist_mlp = zeros(s, DType::FP32, {NUM_VISITS, m, H});
  b.hist_k = zeros(s, DType::FP32, {NUM_VISITS, m, KV_DIM});
  b.hist_v = zeros(s, DType::FP32, {NUM_VISITS, m, KV_DIM});
  return b;
}

std::vector<const float*> Buffers::blocks(int l, int p) const {
  auto f = [](const Tensor& t) { return static_cast<const float*>(t.data); };
  if (l < 1) return {};
  if (l < 7) return {f(B0)};
  if (l < 13) return {f(B0), f(B1)};
  if (l < 19) return {f(B0), f(B1), f(B2[l < LOOP_END ? p : 2])};
  return {f(B0), f(B1), f(B2[2]), f(E[2])};
}

}  // namespace ingine::model
