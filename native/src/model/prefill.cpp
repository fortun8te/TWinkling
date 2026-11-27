// model.Model.prefill for one chunk of T tokens, wired as fast.FastDecoder.decode (block states B0, B1, B2[p],
// E[p], partial / write pairs, message mixes) with T rows instead of M:
//   embed -> per visit: [message mix] -> attention read (mix) [-> message gates] -> taps -> projections
//   (cuBLASLt over the BF16 rows, FP8 rows dequantized per visit as weights.dequant) -> attn_prep (q/rel post-
//   processing, key/value convs, cache or sliding rows) -> rel_logits -> attention (+ gate) -> wo -> attention
//   conv -> MLP read (mix) [-> gate] -> router GEMM + top-8 -> grouped experts -> combine -> MLP conv
//   -> final read (mix) -> taps -> head of the last row.
// The mixes and the embedding are ModelCore's decode kernels (per-row CTAs, any row count). Ablation of the
// attention / MLP writes happens inside the mixes that consume them, as in the decode step.
#include "model/prefill.h"
#include "kernels/moe/moe.h"

#include "core/device.h"
#include "ingine/generator.h"
#include "kernels/dense/dense.h"
#include "kernels/prefill/prefill.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ingine::model {
namespace pk = kernels::prefill;
namespace dense = kernels::dense;

namespace {

constexpr int H = HIDDEN;

const float* f32p(const Tensor& t) { return static_cast<const float*>(t.data); }
const __nv_bfloat16* bf16p(const Tensor& t) { return static_cast<const __nv_bfloat16*>(t.data); }
const __nv_bfloat16* bf16p(const void* p) { return static_cast<const __nv_bfloat16*>(p); }

// fast.Buffers.blocks
std::vector<const float*> blocks_of(int l, int p, const float* B0, const float* B1, float* const* B2, float* const* E) {
  if (l < 1) return {};
  if (l < 7) return {B0};
  if (l < 13) return {B0, B1};
  if (l < 19) return {B0, B1, B2[l < LOOP_END ? p : 2]};
  return {B0, B1, B2[2], E[2]};
}

}  // namespace

struct Prefiller::ExpertTransfer {
  cudaStream_t stream = nullptr;
  cudaEvent_t ready[2] = {nullptr, nullptr};
  cudaEvent_t consumed[2] = {nullptr, nullptr};

  ~ExpertTransfer() {
    if (stream) cudaStreamSynchronize(stream);
    for (int i = 0; i < 2; ++i) {
      if (ready[i]) cudaEventDestroy(ready[i]);
      if (consumed[i]) cudaEventDestroy(consumed[i]);
    }
    if (stream) cudaStreamDestroy(stream);
  }
};

Prefiller::Prefiller(const Weights& w, const Ablation& abl, int max_chunk, cudaStream_t stream, int stage_mib,
                     bool overlap)
    : w_(w), abl_(abl), max_chunk_(max_chunk), stream_(stream), gemm_(std::make_unique<pk::Gemm>()) {
  bool tiered = false;
  for (int l = 0; l < LAYERS; ++l) tiered |= w.experts_keep[l] > w.experts_hot[l];
  if (stage_mib > 0 && tiered) stage_bytes_ = std::size_t(stage_mib) << 20;
  if (w.layers[0].experts.gguf()) {
    const std::size_t E = EXPERTS, R13 = 2 * INTER, R2 = HIDDEN;
    gguf_nv_ = DeviceBuffer(E * R13 * (HIDDEN / 2) + E * R13 * (HIDDEN / 16) + E * R2 * (INTER / 2) +
                            E * R2 * (INTER / 16) + 3 * E * 4 + 1024);
  }
  if (max_chunk < 1) throw std::invalid_argument("Prefiller: max_chunk must be positive");
  const int L = WINDOW + max_chunk;
  kpos_rows_ = (L + pk::kAttnKeyTile - 1) / pk::kAttnKeyTile * pk::kAttnKeyTile;
  if (kpos_rows_ / pk::kAttnKeyTile > pk::kAttnMaxTiles)
    throw std::invalid_argument("Prefiller: max_chunk above " +
                                std::to_string(pk::kAttnMaxTiles * pk::kAttnKeyTile - WINDOW));
  const std::size_t C = std::size_t(max_chunk);
  // Persistent zero-padded sliding buffers stay in storage_ (never overwritten by decode lending).
  auto alloc_persist = [&](std::size_t bytes) {
    storage_.emplace_back(bytes);
    return storage_.back().p;
  };
  krows_ = static_cast<__nv_bfloat16*>(alloc_persist(std::size_t(kpos_rows_) * KV_DIM * 2));
  vrows_ = static_cast<__nv_bfloat16*>(alloc_persist(std::size_t(kpos_rows_) * KV_DIM * 2));
  kpos_ = static_cast<int*>(alloc_persist(std::size_t(kpos_rows_) * 4));
  bounds_ = static_cast<int4*>(alloc_persist(std::size_t(kpos_rows_ / pk::kAttnKeyTile) * sizeof(int4)));
  for (auto& b : storage_) b.fill(0);  // gathered sliding rows past ring + chunk: never keys, kept finite

  // All per-chunk scratch + stage_ live in one contiguous arena_ (idle during decode, lent for cold experts).
  const std::size_t rel_elements = C * HEADS * REL_EXTENT;
  const std::size_t projection_elements = C * PROJ_ROWS_FULL;
  const std::size_t q_elements = C * HEADS * HEAD_DIM;
  const std::size_t rel16_elements = C * HEADS * D_REL;
  const std::size_t attn_out_elements = C * HEADS * HEAD_DIM;
  const std::size_t phase2_elements =
      rel_elements + projection_elements + q_elements + rel16_elements + attn_out_elements;

  const std::size_t mixf_elements = C * 3 * 2 * H;  // 3*C*H floats = 6*C*H bf16
  const std::size_t gate_inputs = C * 3 * 2 * H;
  const std::size_t gate_outputs = C * 4 * H;       // two FP32 rows per token, counted in BF16 elements
  const std::size_t phase1_elements = mixf_elements + gate_inputs + gate_outputs;

  const std::size_t expert_activations = C * TOP_K * INTER;
  const std::size_t expert_outputs = C * TOP_K * H;
  const std::size_t moe_sum_elements = C * H;
  const std::size_t phase3_elements = expert_activations + expert_outputs + moe_sum_elements;
  static_assert(H % 2 == 0);
  const std::size_t phase_elements = std::max({phase1_elements, phase2_elements, phase3_elements});

  const bool fp8_dense_env = [] {
    const char* e = std::getenv("INGINE_PREFILL_FP8_DENSE");
    return !e || std::atoi(e) != 0;
  }();
  fp8_dense_ = w.dense_fp8 && fp8_dense_env;
  const std::size_t wscratch_elements =
      (w.dense_fp8 && !fp8_dense_) ? std::max(std::size_t(PROJ_ROWS_FULL) * H, std::size_t(H) * HEADS * HEAD_DIM) : 0;
  const std::size_t dense_q_bytes =
      std::max(std::size_t(3) * C * H, C * HEADS * HEAD_DIM) +
      ((std::max(std::size_t(3) * C * H, C * HEADS * HEAD_DIM) / 32 + 255) / 256) * 256 + 256;
  const std::size_t moe_up_q_bytes = C * (H / 2) + ((C * (H / 16) + 255) / 256) * 256 + C * 4 + 512;
  const std::size_t moe_dn_q_bytes =
      C * TOP_K * (INTER / 2) + ((C * TOP_K * (INTER / 16) + 255) / 256) * 256 + C * TOP_K * 4 + 512;
  qbuf_bytes_ = std::max({dense_q_bytes, moe_up_q_bytes, moe_dn_q_bytes});

  auto align256 = [](std::size_t n) { return (n + 255) / 256 * 256; };
  std::size_t arena_total = 0;
  auto plan_bytes = [&](std::size_t bytes) { arena_total += align256(bytes); };
  for (int i = 0; i < 9; ++i) plan_bytes(C * H * 4);  // B0, B1, B2[0..1], E[0..1], hid, part, y
  plan_bytes(3 * C * H * 2);                          // xin
  plan_bytes(C * H * 2);                              // xm
  plan_bytes(C * H * 2);                              // xf
  plan_bytes(4 * C * H * 4);                          // msgs
  plan_bytes(phase_elements * 2);                     // phase
  plan_bytes(3 * C * GATE_RANK * 2);                  // g16
  plan_bytes(C * EXPERTS * 4);                        // router_logits
  plan_bytes(C * TOP_K * 4);                          // wts
  plan_bytes(C * TOP_K * 4);                          // idx
  plan_bytes(EXPERTS * 4);                            // counts
  plan_bytes(4);                                      // tiles_n
  plan_bytes(C * TOP_K * 4);                          // pair_token
  plan_bytes(C * TOP_K * 4);                          // pair_dst
  plan_bytes(std::size_t(pk::moe_tiles_capacity(max_chunk)) * sizeof(int4));  // tiles
  if (wscratch_elements) plan_bytes(wscratch_elements * 2);
  plan_bytes(qbuf_bytes_);                            // qbuf
  plan_bytes(C * 8);                                  // ids
  if (stage_bytes_) plan_bytes(stage_bytes_);         // stage

  arena_ = DeviceBuffer(arena_total);
  arena_.fill(0);
  char* cur = static_cast<char*>(arena_.p);
  auto alloc = [&](std::size_t bytes) -> void* {
    void* p = cur;
    cur += align256(bytes);
    return p;
  };
  auto f32 = [&](std::size_t n) { return static_cast<float*>(alloc(n * 4)); };
  auto bf16 = [&](std::size_t n) { return static_cast<__nv_bfloat16*>(alloc(n * 2)); };
  auto i32 = [&](std::size_t n) { return static_cast<int*>(alloc(n * 4)); };

  B0_ = f32(C * H), B1_ = f32(C * H);
  // B2_[2] aliases B2_[0] and E_[2] aliases E_[0] (pass 0's blocks are never read after pass 1);
  // y2_ aliases y_ (attention write and MLP write have disjoint lifetimes).
  B2_[0] = f32(C * H), B2_[1] = f32(C * H), B2_[2] = B2_[0];
  E_[0] = f32(C * H), E_[1] = f32(C * H), E_[2] = E_[0];
  hid_ = f32(C * H), part_ = f32(C * H), y_ = f32(C * H), y2_ = y_;
  xin_ = bf16(3 * C * H), xm_ = bf16(C * H), xf_ = bf16(C * H);
  msgs_ = f32(4 * C * H);
  auto* phase = bf16(phase_elements);

  mixf_ = reinterpret_cast<float*>(phase);
  gin_ = phase + mixf_elements;
  fpre_ = reinterpret_cast<float*>(gin_ + gate_inputs);
  gpre_ = fpre_ + C * H;
  mixm_ = reinterpret_cast<float*>(phase);  // MLP mix/gates run before MoE

  rel_ = phase;
  proj_ = rel_ + rel_elements;
  q_ = proj_ + projection_elements;
  rel16_ = q_ + q_elements;
  attn_out_ = rel16_ + rel16_elements;
  wo_out_ = rel_;  // rel_ is dead once attention finishes

  act_ = phase;
  down_ = act_ + expert_activations;
  moe_sum_ = down_ + expert_outputs;

  g16_ = bf16(3 * C * GATE_RANK);
  router_logits_ = f32(C * EXPERTS), wts_ = f32(C * TOP_K);
  idx_ = i32(C * TOP_K);
  counts_ = i32(EXPERTS), tiles_n_ = i32(1), pair_token_ = i32(C * TOP_K), pair_dst_ = i32(C * TOP_K);
  tiles_ = static_cast<int4*>(alloc(std::size_t(pk::moe_tiles_capacity(max_chunk)) * sizeof(int4)));
  wscratch_ = wscratch_elements ? bf16(wscratch_elements) : nullptr;
  qbuf_ = static_cast<uint8_t*>(alloc(qbuf_bytes_));
  ids_ = static_cast<std::int64_t*>(alloc(C * 8));
  stage_ptr_ = stage_bytes_ ? alloc(stage_bytes_) : nullptr;
  if (stage_ptr_ && overlap) {
    stage_slots_ = stage_bytes_ >= (128u << 20) ? 2 : 1;
    transfer_ = std::make_unique<ExpertTransfer>();
    CUDA_CHECK(cudaStreamCreateWithFlags(&transfer_->stream, cudaStreamNonBlocking));
    for (int i = 0; i < 2; ++i) {
      CUDA_CHECK(cudaEventCreateWithFlags(&transfer_->ready[i], cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreateWithFlags(&transfer_->consumed[i], cudaEventDisableTiming));
    }
  }
}

Prefiller::~Prefiller() = default;

void Prefiller::sync_transfer_after_stream(cudaStream_t s) {
  if (!transfer_) return;
  for (int i = 0; i < stage_slots_; ++i) CUDA_CHECK(cudaEventRecord(transfer_->consumed[i], s));
}

std::size_t Prefiller::device_bytes() const {
  std::size_t n = arena_.bytes;
  for (const auto& b : storage_) n += b.bytes;
  return n;
}

const float* const* Prefiller::block_table(const std::vector<const float*>& blocks) {
  auto it = tables_.find(blocks);
  if (it == tables_.end()) {
    DeviceBuffer b(std::max<std::size_t>(blocks.size(), 1) * sizeof(void*));
    if (!blocks.empty()) b.copy_from_host(blocks.data(), blocks.size() * sizeof(void*));
    it = tables_.emplace(blocks, std::move(b)).first;
  }
  return static_cast<const float* const*>(it->second.p);
}

void Prefiller::run(State& st, const std::int32_t* tokens, int T, int pos0, float* logits, Tensor* taps,
                    std::span<const EmbedRows> rows) {
  if (T < 1 || T > max_chunk_) throw std::invalid_argument("Prefiller::run: T outside 1..max_chunk");
  if (pos0 < 0 || pos0 + T > st.max_ctx)
    throw std::invalid_argument("context " + std::to_string(pos0 + T) + " exceeds max_ctx " +
                                std::to_string(st.max_ctx));
  const Weights& w = w_;
  const cudaStream_t s = stream_;
  const std::size_t slab = std::size_t(T) * H;  // one [T, H] slab of a [G, T, H] buffer
  const float* attn_dirs = f32p(abl_.attn_dirs);
  const float* attn_w = f32p(abl_.attn_w);
  const float* mlp_dirs = f32p(abl_.mlp_dirs);
  const float* mlp_w = f32p(abl_.mlp_w);
  __nv_bfloat16* tap_out = taps ? static_cast<__nv_bfloat16*>(taps->data) : nullptr;

  auto mix = [&](const std::vector<const float*>& blocks, const float* base, const float* write, const float* abl_r,
                 const float* abl_w, float* partial_out, const float* queries, int Q, const float* norm_w,
                 __nv_bfloat16* out_bf16, float* out_f32) {
    dense::MixParams p;
    p.blocks = blocks.empty() ? nullptr : block_table(blocks);
    p.nblocks = int(blocks.size());
    p.base = base;
    p.write = write;
    p.abl_r = write ? abl_r : nullptr;
    p.abl_d = write && abl_.detect ? abl_r + std::size_t(NUM_VISITS) * H : nullptr;  // the detector of the same visit
    p.abl_w = write ? abl_w : nullptr;
    p.partial_out = partial_out;
    p.queries = queries;
    p.Q = Q;
    p.norm_w = norm_w;
    p.out_bf16 = out_bf16;
    p.out_f32 = out_f32;
    p.M = T;
    dense::mix(p, false, s);
  };
  // B rows of a Dense matrix as BF16: the matrix itself, or its FP8 rows dequantized (weights.dequant)
  auto bf16_rows = [&](const Dense& d) -> const __nv_bfloat16* {
    if (!d.fp8()) return bf16p(d.w);
    dense::dequant_rows(d.w.data, f32p(d.scale), d.rows(), d.cols(), wscratch_, s);
    return wscratch_;
  };
  // message_gate for G gates: h_g, msg_g fp32 [T, H] -> out bf16 [G, T, H] = bf16(rms_norm(gated) * norm_w[g])
  static const bool fused_gate = [] {
    const char* e = std::getenv("INGINE_PREFILL_FUSED_GATE");
    return !e || std::atoi(e) != 0;
  }();
  auto gates = [&](const pk::GateRows& rows, int G, const Tensor& down, const Tensor& forget16, const Tensor& gain16,
                   const float* norm_w, __nv_bfloat16* out) {
    if (fused_gate) {
      pk::gate_down_fused(rows, G, T, bf16p(down), g16_, s);
      for (int g = 0; g < G; ++g) {
        const __nv_bfloat16* gg = g16_ + std::size_t(g) * T * GATE_RANK;
        gemm_->run(gg, GATE_RANK, bf16p(forget16) + std::size_t(g) * H * GATE_RANK, GATE_RANK, fpre_, H, true, T, H,
                   GATE_RANK, s);
        gemm_->run(gg, GATE_RANK, bf16p(gain16) + std::size_t(g) * H * GATE_RANK, GATE_RANK, gpre_, H, true, T, H,
                   GATE_RANK, s);
        pk::gate_apply(rows.h[g], rows.msg[g], fpre_, gpre_, norm_w + std::size_t(g) * H, T,
                       out + std::size_t(g) * slab, s);
      }
      return;
    }
    pk::gate_inputs(rows, G, T, gin_, s);
    for (int g = 0; g < G; ++g) {
      const __nv_bfloat16* gd = bf16p(down) + std::size_t(g) * GATE_RANK * 2 * H;
      gemm_->run(gin_ + std::size_t(g) * T * 2 * H, 2 * H, gd, 2 * H, g16_, GATE_RANK, false, T, GATE_RANK, 2 * H, s);
      gemm_->run(g16_, GATE_RANK, bf16p(forget16) + std::size_t(g) * H * GATE_RANK, GATE_RANK, fpre_, H, true, T, H,
                 GATE_RANK, s);
      gemm_->run(g16_, GATE_RANK, bf16p(gain16) + std::size_t(g) * H * GATE_RANK, GATE_RANK, gpre_, H, true, T, H,
                 GATE_RANK, s);
      pk::gate_apply(rows.h[g], rows.msg[g], fpre_, gpre_, norm_w + std::size_t(g) * H, T, out + std::size_t(g) * slab,
                     s);
    }
  };

  auto stage_experts = [&](const Layer& layer, pk::MoeExperts& me, cudaStream_t copy_stream, int slot) {
    const int hot = w.experts_hot[layer.idx], keep = w.experts_keep[layer.idx];
    if (!stage_ptr_ || keep <= hot) return;
    const Experts& experts = layer.experts;
    const std::size_t b13 = experts.w13.bytes() / EXPERTS, b13s = experts.w13_scale.bytes() / EXPERTS;
    const std::size_t b2 = experts.w2.bytes() / EXPERTS, b2s = experts.w2_scale.bytes() / EXPERTS;
    const std::size_t bytes_per_expert = b13 + b13s + b2 + b2s;
    const std::size_t slot_bytes = stage_bytes_ / std::size_t(stage_slots_);
    if (slot_bytes <= 1024) return;
    const int n = std::min<int>(keep - hot, int((slot_bytes - 1024) / bytes_per_expert));
    if (n == 0) return;

    char* dst = static_cast<char*>(stage_ptr_) + std::size_t(slot) * slot_bytes;
    auto copy = [&](const Tensor& tensor, std::size_t bytes, bool pad_zero) {
      char* start = dst;
      CUDA_CHECK(cudaMemcpyAsync(start, static_cast<const char*>(tensor.data) + std::size_t(hot) * bytes,
                                 std::size_t(n) * bytes, cudaMemcpyDefault, copy_stream));
      if (pad_zero) CUDA_CHECK(cudaMemsetAsync(start + std::size_t(n) * bytes, 0, 16, copy_stream));
      dst += ((std::size_t(n) * bytes + (pad_zero ? 16 : 0) + 255) / 256) * 256;
      return start;
    };
    me.stage_w13 = copy(experts.w13, b13, false);
    me.stage_w13_scale = copy(experts.w13_scale, b13s, true);
    me.stage_w2 = copy(experts.w2, b2, false);
    me.stage_w2_scale = copy(experts.w2_scale, b2s, true);
    me.stage_first = hot;
    me.stage_n = n;
  };

  auto align256 = [](std::size_t n) { return (n + 255) / 256 * 256; };
  auto make_moe_q = [&](int rows, int K) {
    pk::MoeQuantScratch qs;
    const std::size_t v_bytes = align256(std::size_t(rows) * (K / 2));
    const std::size_t s_bytes = align256(std::size_t(rows) * (K / 16));
    qs.xq = qbuf_;
    qs.xscale = qbuf_ + v_bytes;
    qs.xouter = reinterpret_cast<float*>(qbuf_ + v_bytes + s_bytes);
    return qs;
  };

  pk::widen_ids(tokens, T, ids_, IMAGE_PLACEHOLDER_ID, s);
  dense::embed(ids_, T, bf16p(w.embed), f32p(w.embed_norm), B0_, false, s);
  // image tokens: the vision tower's rows take the place of the placeholder's embedding (model.forward's
  // masked_scatter into the normed embeddings); every later read of the embedding block sees them
  for (const EmbedRows& r : rows) {
    if (r.row < 0 || r.count < 0 || r.row + r.count > T) throw std::invalid_argument("Prefiller::run: embedding rows outside the chunk");
    pk::put_rows(r.src, r.count, B0_ + std::size_t(r.row) * H, s);
  }
  int prev_v = -1;
  int tap = 0;
  int stage_seq = 0;

  for (int v = 0; v < NUM_VISITS; ++v) {
    const int l = VISITS[v].layer, p = VISITS[v].pass;
    const Layer& layer = w.layers[l];
    const Experts& ex = layer.experts;
    pk::MoeExperts me{ex.fmt == ExpertFormat::NVFP4 ? 0 : 1, ex.w13.data, ex.w13_scale.data,
                      f32p(ex.w13_factor), ex.w2.data, ex.w2_scale.data, f32p(ex.w2_factor)};
    me.w13_scale_base = ex.w13_scale_base;
    me.w2_scale_base = ex.w2_scale_base;
    const int slot = stage_seq % stage_slots_;
    if (transfer_ && w.experts_keep[l] > w.experts_hot[l]) {
      CUDA_CHECK(cudaStreamWaitEvent(transfer_->stream, transfer_->consumed[slot], 0));
      stage_experts(layer, me, transfer_->stream, slot);
      if (me.stage_n) CUDA_CHECK(cudaEventRecord(transfer_->ready[slot], transfer_->stream));
    }
    const bool msgs = is_loop_layer(l) && p > 0;
    const bool boundary = l % BLOCK == 0;
    const std::vector<const float*> blocks = blocks_of(l, p, B0_, B1_, B2_, E_);

    // ---- hidden entering this visit (the previous MLP write applied), stored where it is a block
    float* hid;
    const float* base;
    const float* write;
    if (v == 0) {
      hid = B0_, base = B0_, write = nullptr;
    } else if (l == LOOP_START && p > 0) {
      hid = B1_, base = B1_, write = nullptr;
    } else {
      hid = l == 6 ? B1_ : l == 12 ? B2_[p] : l == 18 ? E_[2] : hid_;
      base = part_, write = y2_;
    }
    const int pv = prev_v >= 0 ? prev_v : 0;
    const float* r_prev = mlp_dirs + std::size_t(pv) * H;
    const float* w_prev = mlp_w + 4 * std::size_t(pv);
    if (msgs) {
      const int i = l - LOOP_START;
      const float* src_base = i == 0 ? part_ : E_[p - 1];
      const float* src_write = i == 0 ? y2_ : nullptr;
      mix({B2_[p - 1]}, src_base, src_write, r_prev, w_prev, i == 0 ? E_[p - 1] : nullptr,
          f32p(w.message_queries) + std::size_t(i) * 4 * H, 4, nullptr, nullptr, msgs_);
      if (i == 0) write = nullptr;
    }
    // ---- attention read(s)
    const float* queries = layer.attn_query.data ? f32p(layer.attn_query) : nullptr;
    const int nq = layer.attn_query.data ? layer.attn_query.ne[1] : 1;
    float* partial_out = hid != base ? hid : nullptr;
    if (msgs) {
      mix(blocks, base, write, r_prev, w_prev, partial_out, queries, nq, nullptr, nullptr, mixf_);
      pk::GateRows rows;
      for (int g = 0; g < 3; ++g) rows.h[g] = mixf_ + (layer.full ? g : 0) * slab, rows.msg[g] = msgs_ + g * slab;
      gates(rows, 3, layer.qkv_gate_down, layer.qkv_gate_forget16, layer.qkv_gate_gain16, f32p(layer.attn_norm3),
            xin_);
    } else {
      mix(blocks, base, write, r_prev, w_prev, partial_out, l > 0 ? queries : nullptr, l > 0 ? nq : 1,
          f32p(layer.attn_norm_mix), xin_, nullptr);
    }
    if (tap < 3 && v == TAP_VISITS[tap]) {
      if (tap_out) pk::store_tap(hid, T, tap_out + std::size_t(tap) * H, std::int64_t(NUM_TAPS) * H, s);
      ++tap;
    }

    // ---- projections (fast.proj_layout): one GEMM for a single input, else per input segment
    const int R = proj_rows(layer.full);
    const bool single = !layer.full && !msgs;
    if (fp8_dense_ && layer.wproj.fp8()) {
      const int qrows = single ? T : 3 * T;
      uint8_t* xq = qbuf_;
      uint8_t* xs = qbuf_ + align256(std::size_t(qrows) * H);
      pk::quantize_mxfp8(xin_, qrows, H, xq, xs, s);
      const auto* wp8 = static_cast<const uint8_t*>(layer.wproj.w.data);
      const float* ws = f32p(layer.wproj.scale);
      if (single) {
        pk::Fp8GemmProduct prod{wp8, ws, xq, xs, proj_, R, R};
        pk::gemm_fp8_batch(&prod, 1, T, H, s);
      } else {
        struct Part {
          int begin, end, input;
        };
        const Part parts[] = {{SEG_BEGIN[SEG_Q], SEG_BEGIN[SEG_K], 0},
                              {SEG_BEGIN[SEG_K], SEG_BEGIN[SEG_V], 1},
                              {SEG_BEGIN[SEG_V], SEG_BEGIN[SEG_K_DYN], 2},
                              {SEG_BEGIN[SEG_K_DYN], SEG_BEGIN[SEG_V_DYN], 1},
                              {SEG_BEGIN[SEG_V_DYN], SEG_BEGIN[NUM_SEGS], 2}};
        pk::Fp8GemmProduct prods[5];
        int np = 0;
        for (const Part& part : parts) {
          if (part.begin >= R) break;
          prods[np++] = pk::Fp8GemmProduct{
              wp8 + std::size_t(part.begin) * H, ws + part.begin,
              xq + std::size_t(part.input) * slab, xs + std::size_t(part.input) * (slab / 32),
              proj_ + part.begin, part.end - part.begin, R};
        }
        pk::gemm_fp8_batch(prods, np, T, H, s);
      }
    } else {
      const __nv_bfloat16* wp = bf16_rows(layer.wproj);
      if (single) {
        gemm_->run(xin_, H, wp, H, proj_, R, false, T, R, H, s);
      } else {
        struct Part {
          int begin, end, input;
        };
        const Part parts[] = {{SEG_BEGIN[SEG_Q], SEG_BEGIN[SEG_K], 0},
                              {SEG_BEGIN[SEG_K], SEG_BEGIN[SEG_V], 1},
                              {SEG_BEGIN[SEG_V], SEG_BEGIN[SEG_K_DYN], 2},
                              {SEG_BEGIN[SEG_K_DYN], SEG_BEGIN[SEG_V_DYN], 1},
                              {SEG_BEGIN[SEG_V_DYN], SEG_BEGIN[NUM_SEGS], 2}};
        for (const Part& part : parts) {
          if (part.begin >= R) break;
          gemm_->run(xin_ + std::size_t(part.input) * slab, H, wp + std::size_t(part.begin) * H, H, proj_ + part.begin,
                     R, false, T, part.end - part.begin, H, s);
        }
      }
    }

    // ---- attention
    pk::AttnPrepParams ap;
    ap.proj = proj_, ap.ld = R, ap.T = T, ap.pos0 = pos0, ap.full = layer.full;
    ap.q_scale = f32p(layer.q_scale), ap.r_scale = f32p(layer.r_scale);
    ap.k_w = f32p(layer.full ? layer.k_static : layer.k_conv);
    ap.v_w = f32p(layer.full ? layer.v_static : layer.v_conv);
    ap.conv_k = st.conv_k_at(v), ap.conv_v = st.conv_v_at(v);
    ap.q = q_, ap.rel = rel16_;
    pk::AttnParams at;
    at.q = q_, at.T = T, at.pos0 = pos0, at.rel = rel_, at.proj = proj_, at.ld = R, at.out = attn_out_;
    if (layer.full) {
      const int fi = FULL_INDEX[v];
      const Tensor k_cache = st.k_full_at(fi);
      void* kc = k_cache.data;
      const std::size_t off = std::size_t(pos0) * (k_cache.bytes() / st.max_ctx);
      const Tensor v_cache = st.v_full_at(fi);
      void* vc = v_cache.data;
      const std::size_t v_off = std::size_t(pos0) * (v_cache.bytes() / st.max_ctx);
      ap.k_out = static_cast<char*>(kc) + off, ap.v_out = static_cast<char*>(vc) + v_off, ap.out_fp8 = st.kv_fp8;
      ap.out_v4 = st.kv_v4;
      ap.out_k4 = st.kv_k4;
      pk::attn_prep(ap, s);
      at.k = kc, at.v = vc, at.kv_fp8 = st.kv_fp8, at.kpos = nullptr, at.bounds = nullptr, at.L = pos0 + T;
      at.window = 0;
      at.kv_v4 = st.kv_v4;
      at.kv_k4 = st.kv_k4;
    } else {
      const int ri = RING_INDEX[v];
      void* k_ring = st.k_ring_at(ri).data;
      void* v_ring = st.v_ring_at(ri).data;
      std::int64_t* ring_pos = st.ring_pos_at(ri);
      const bool rfp8 = st.ring_fp8;
      // keys = cat(ring, chunk) as model.prefill reads them, before the ring takes the chunk's last rows
      pk::ring_gather(k_ring, v_ring, ring_pos, pos0, T, krows_, vrows_, kpos_, kpos_rows_, rfp8, s);
      const std::size_t row_off = std::size_t(WINDOW) * KV_DIM * (rfp8 ? 1 : 2);
      ap.k_out = reinterpret_cast<char*>(krows_) + row_off;
      ap.v_out = reinterpret_cast<char*>(vrows_) + row_off;
      ap.out_fp8 = rfp8;
      pk::attn_prep(ap, s);
      pk::ring_update(krows_, vrows_, pos0, T, k_ring, v_ring, ring_pos, rfp8, s);
      pk::key_tile_bounds(kpos_, WINDOW + T, bounds_, s);
      at.k = krows_, at.v = vrows_, at.kv_fp8 = rfp8, at.kpos = kpos_, at.bounds = bounds_, at.L = WINDOW + T;
      at.window = WINDOW;
    }
    pk::conv_state_update(st.conv_k_at(v), proj_ + SEG_BEGIN[SEG_K], R, T, KV_DIM, s);
    pk::conv_state_update(st.conv_v_at(v), proj_ + SEG_BEGIN[SEG_V], R, T, KV_DIM, s);
    pk::rel_logits(rel16_, f32p(layer.rel_proj), T, rel_, s);
    pk::attention(at, s);

    // ---- output projection + post-attention conv
    if (fp8_dense_ && layer.wo.fp8()) {
      constexpr int K_WO = HEADS * HEAD_DIM;
      uint8_t* xq = qbuf_;
      uint8_t* xs = qbuf_ + align256(std::size_t(T) * K_WO);
      pk::quantize_mxfp8(attn_out_, T, K_WO, xq, xs, s);
      pk::Fp8GemmProduct prod{layer.wo.w.data, f32p(layer.wo.scale), xq, xs, wo_out_, H, H};
      pk::gemm_fp8_batch(&prod, 1, T, K_WO, s);
    } else {
      const __nv_bfloat16* wo = bf16_rows(layer.wo);
      gemm_->run(attn_out_, HEADS * HEAD_DIM, wo, HEADS * HEAD_DIM, wo_out_, H, false, T, H, HEADS * HEAD_DIM, s);
    }
    pk::static_conv(wo_out_, H, T, H, f32p(layer.attn_conv), st.conv_attn_at(v), y_, s);
    pk::conv_state_update(st.conv_attn_at(v), wo_out_, H, T, H, s);

    // ---- MLP read: this visit's attention write, ablated, on top of hid (or a fresh block)
    std::vector<const float*> mlp_blocks = blocks;
    if (boundary) mlp_blocks.push_back(hid);
    const float* mbase = boundary ? nullptr : hid;
    if (msgs) {
      mix(mlp_blocks, mbase, y_, attn_dirs + std::size_t(v) * H, attn_w + 4 * std::size_t(v), part_, f32p(layer.mlp_query), 1, nullptr,
          nullptr, mixm_);
      pk::GateRows rows;
      rows.h[0] = mixm_, rows.msg[0] = msgs_ + 3 * slab;
      gates(rows, 1, layer.mlp1_gate_down, layer.mlp1_gate_forget16, layer.mlp1_gate_gain16, f32p(layer.mlp_norm1),
            xm_);
    } else {
      mix(mlp_blocks, mbase, y_, attn_dirs + std::size_t(v) * H, attn_w + 4 * std::size_t(v), part_, f32p(layer.mlp_query), 1,
          f32p(layer.mlp_norm1), xm_, nullptr);
    }

    // ---- routed experts (model._moe_grouped) + post-MLP conv
    gemm_->run(xm_, H, bf16p(layer.router_bf16), H, router_logits_, EXPERTS, true, T, EXPERTS, H, s);
    CUDA_CHECK(cudaMemsetAsync(counts_, 0, EXPERTS * sizeof(int), s));
    pk::moe_topk(router_logits_, f32p(layer.router_bias), T, idx_, wts_, counts_, s);
    const pk::MoeWork work{counts_, tiles_n_, tiles_, pair_token_, pair_dst_};
    pk::moe_route(idx_, T, work, s);
    if (!transfer_) stage_experts(layer, me, s, 0);
    else if (me.stage_n) CUDA_CHECK(cudaStreamWaitEvent(s, transfer_->ready[slot], 0));
    if (ex.gguf()) {
      // transcode this layer's GGUF experts into the NVFP4 scratch the grouped GEMMs read
      const std::size_t E = EXPERTS, R13 = 2 * INTER, R2 = H;
      char* base = static_cast<char*>(gguf_nv_.p);
      char* c13 = base;
      char* s13 = c13 + E * R13 * (H / 2);
      char* c2 = s13 + E * R13 * (H / 16);
      char* s2 = c2 + E * R2 * (INTER / 2);
      float* f13 = reinterpret_cast<float*>((reinterpret_cast<std::uintptr_t>(s2 + E * R2 * (INTER / 16)) + 255) & ~std::uintptr_t(255));
      float* f2 = f13 + E;
      void* amax = f2 + E;
      kernels::moe::gguf_to_nvfp4(ex.g13_type, ex.g13, ex.g13_row, int(R13), H, int(E), true, amax, c13, s13, f13, s);
      kernels::moe::gguf_to_nvfp4(ex.g2_type, ex.g2, ex.g2_row, int(R2), INTER, int(E), false, amax, c2, s2, f2, s);
      me = pk::MoeExperts{0, c13, s13, f13, c2, s2, f2};
    }
    pk::MoeQuantScratch q_up = make_moe_q(T, H);
    pk::MoeQuantScratch q_dn;
    if (pk::moe_fused_actq_active(me.fmt, T)) {
      auto* act_q = reinterpret_cast<uint8_t*>(act_);
      const std::size_t q_bytes = align256(std::size_t(T) * TOP_K * (INTER / 2));
      q_up.out_xq = act_q;
      q_up.out_xscale = act_q + q_bytes;
      q_dn.xq = act_q;
      q_dn.xscale = act_q + q_bytes;
      q_dn.xouter = nullptr;
      q_dn.xouter_const = pk::kFusedActOuter;
      q_dn.prequantized = true;
    } else {
      q_dn = make_moe_q(T * TOP_K, INTER);
    }
    pk::moe_gate_up(me, xm_, T, work, act_, s, &q_up);
    pk::moe_down(me, act_, T, work, down_, s, &q_dn);
    if (transfer_ && me.stage_n) {
      CUDA_CHECK(cudaEventRecord(transfer_->consumed[slot], s));
      ++stage_seq;
    }
    if (stats_) pk::moe_stats(idx_, wts_, down_, T, stats_ + std::size_t(v) * EXPERTS * 3, s);
    pk::moe_combine(down_, wts_, T, moe_sum_, s);
    pk::static_conv(moe_sum_, H, T, H, f32p(layer.mlp_conv), st.conv_mlp_at(v), y2_, s);
    pk::conv_state_update(st.conv_mlp_at(v), moe_sum_, H, T, H, s);
    prev_v = v;
  }

  // ---- final read, taps, head of the last row
  if (logits != nullptr || tap_out != nullptr) {
    mix({B0_, B1_, B2_[2], E_[2]}, part_, y2_, mlp_dirs + std::size_t(prev_v) * H, mlp_w + 4 * std::size_t(prev_v),
        nullptr, f32p(w.final_query), 1, f32p(w.final_norm), xf_, nullptr);
  }
  if (tap_out)
    CUDA_CHECK(cudaMemcpy2DAsync(tap_out + std::size_t(NUM_TAPS - 1) * H, std::size_t(NUM_TAPS) * H * 2, xf_,
                                 std::size_t(H) * 2, std::size_t(H) * 2, T, cudaMemcpyDeviceToDevice, s));
  if (logits != nullptr) {
    if (w.unembed_q4_codes.data) {
      dense::head_q4_certified(xf_ + std::size_t(T - 1) * H, 1,
                               dense::Q4Matrix{w.unembed_q4_codes.data, w.unembed_q4_scales.data, UNPADDED_VOCAB},
                               dense::Matrix{w.unembed.w.data, f32p(w.unembed.scale), UNPADDED_VOCAB}, logits, s);
    } else {
      pk::head_logits(xf_ + std::size_t(T - 1) * H, w.unembed.w.data, f32p(w.unembed.scale), UNPADDED_VOCAB, logits,
                      s);
    }
  }
  CUDA_CHECK(cudaGetLastError());
  st.length = pos0 + T;
}

}  // namespace ingine::model
