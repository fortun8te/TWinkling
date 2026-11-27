#include "model/dflash2.h"
#include "model/tp_runtime.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/safetensors.h"
#include "kernels/dense/dense.h"
#include "kernels/draft/dflash2_selector.h"
#include "kernels/draft/gemm.h"
#include "kernels/draft/draft.h"
#include "kernels/draft/shortlist.h"
#include "kernels/head/nvfp4_head.h"

#include <math_constants.h>
#include <mma.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace ingine::model {
namespace {
using Bf = __nv_bfloat16;
namespace head = kernels::head;
constexpr int H = 2304, I = 6912, Q = 4096, KV = 1024, W = 2048;
constexpr int D = 128, QHeads = Q / D, KVHeads = KV / D;
constexpr int Group = 16, Groups = H / Group, Coefficients = 4 * Groups;
constexpr int CommitBatch = 128, Threads = 128;
constexpr int Block = DFlash2::block, Proposals = Block - 1;
constexpr int Rank = 256, Vocabulary = 201024;

__global__ void normalize(const Bf* x, const Bf* weight, int width, int heads,
                          int position, Bf* out, int ring, const int* device_position = nullptr) {
  if (device_position) position = *device_position;
  __shared__ float sums[Threads];
  const int row = blockIdx.x / heads;
  const int head_idx = blockIdx.x % heads;
  const int lane = threadIdx.x;
  x += blockIdx.x * width;
  const int slot = ring ? (position + row) % ring : row;
  out += (slot * heads + head_idx) * width;
  float sum = 0;
  for (int i = lane; i < width; i += Threads) {
    const float value = __bfloat162float(x[i]);
    sum += value * value;
  }
  sums[lane] = sum;
  __syncthreads();
  for (int stride = Threads / 2; stride; stride /= 2) {
    if (lane < stride) sums[lane] += sums[lane + stride];
    __syncthreads();
  }
  const float scale = rsqrtf(sums[0] / width + 1e-6f);
  if (position < 0) {
    for (int i = lane; i < width; i += Threads)
      out[i] = __float2bfloat16(__bfloat162float(x[i]) * scale * __bfloat162float(weight[i]));
  } else {
    const int other = (lane + D / 2) % D;
    const float value = __bfloat162float(__float2bfloat16(__bfloat162float(x[lane]) * scale *
                                                        __bfloat162float(weight[lane])));
    const float pair = __bfloat162float(__float2bfloat16(__bfloat162float(x[other]) * scale *
                                                       __bfloat162float(weight[other])));
    float sine, cosine;
    sincosf((position + row) * powf(10000.f, -2.f * (lane % (D / 2)) / D), &sine, &cosine);
    out[lane] = __float2bfloat16(value * cosine + (lane < D / 2 ? -pair : pair) * sine);
  }
}

__global__ void convolution(const Bf* x, const Bf* coeff, const Bf* base, int part,
                            Bf* out, bool residual) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  if (col >= H) return;
  coeff += row * Coefficients + part * 2 * Groups + col / Group;
  base += part * 2 * H + col;
  const int index = row * H + col;
  const Bf a = __hadd(base[0], coeff[0]);
  const Bf b = __hadd(base[H], coeff[Groups]);
  const Bf shifted = row ? x[index - H] : __float2bfloat16(0.f);
  const Bf value = __hadd_rn(__hmul_rn(a, x[index]), __hmul_rn(b, shifted));
  out[index] = residual ? __hadd(out[index], value) : value;
}

__global__ void swiglu(const Bf* gate, const Bf* up, Bf* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const float x = __bfloat162float(gate[i]);
  out[i] = __hmul(__float2bfloat16(x / (1.f + expf(-x))), up[i]);
}

__global__ void convert(const float* x, Bf* out, int size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) out[i] = __float2bfloat16(x[i]);
}

__global__ void widen(const Bf* x, float* out, int size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) out[i] = __bfloat162float(x[i]);
}

__global__ void sum_bf16_pair(const Bf* a, const Bf* b, Bf* out_a, Bf* out_b, int size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    const Bf v = __float2bfloat16_rn(__bfloat162float(a[i]) + __bfloat162float(b[i]));
    out_a[i] = v;
    out_b[i] = v;
  }
}

__global__ void sum_f32_pair_to_bf16(const float* a, const float* b, Bf* out_a, Bf* out_b, int size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    const Bf v = __float2bfloat16_rn(a[i] + b[i]);
    out_a[i] = v;
    out_b[i] = v;
  }
}

// One CTA per query. Warp-cooperative dot products followed by a stable
// softmax and a channel-parallel value reduction.
template <int LocalQHeads, int LocalKVHeads>
__global__ void attend(const Bf* query, const Bf* ring_k, const Bf* ring_v,
                       const Bf* block_k, const Bf* block_v, int position, Bf* out,
                       const int* device_position = nullptr) {
  constexpr int LocalKV = LocalKVHeads * D;
  if (device_position) position = *device_position;
  __shared__ float scores[W + Block];
  __shared__ float reductions[Threads];
  const int tid = threadIdx.x;
  const int qrow = blockIdx.x / LocalQHeads;
  const int kvhead = (blockIdx.x % LocalQHeads) / (LocalQHeads / LocalKVHeads);
  const int begin = max(0, position + qrow - W + 1);
  const int count = position + Block - begin;
  query += blockIdx.x * D;
  for (int i = tid / 32; i < count; i += Threads / 32) {
    const int pos = begin + i;
    const Bf* key = (pos < position ? ring_k + (pos % W) * LocalKV : block_k + (pos - position) * LocalKV)
                   + kvhead * D;
    float dot = 0;
    for (int j = tid % 32; j < D; j += 32)
      dot += __bfloat162float(query[j]) * __bfloat162float(key[j]);
    for (int offset = 16; offset; offset /= 2)
      dot += __shfl_down_sync(0xffffffff, dot, offset);
    if (tid % 32 == 0) scores[i] = dot * rsqrtf(float(D));
  }
  __syncthreads();
  float local_max = -CUDART_INF_F;
  for (int i = tid; i < count; i += Threads) local_max = fmaxf(local_max, scores[i]);
  reductions[tid] = local_max;
  __syncthreads();
  for (int stride = Threads / 2; stride; stride /= 2) {
    if (tid < stride) reductions[tid] = fmaxf(reductions[tid], reductions[tid + stride]);
    __syncthreads();
  }
  const float peak = reductions[0];
  __syncthreads();
  float sum = 0;
  for (int i = tid; i < count; i += Threads) {
    scores[i] = expf(scores[i] - peak);
    sum += scores[i];
  }
  reductions[tid] = sum;
  __syncthreads();
  for (int stride = Threads / 2; stride; stride /= 2) {
    if (tid < stride) reductions[tid] += reductions[tid + stride];
    __syncthreads();
  }
  float value = 0;
  for (int i = 0; i < count; ++i) {
    const int pos = begin + i;
    const Bf* row = pos < position ? ring_v + (pos % W) * LocalKV : block_v + (pos - position) * LocalKV;
    value += scores[i] * __bfloat162float(row[kvhead * D + tid]);
  }
  out[blockIdx.x * D + tid] = __float2bfloat16(value / reductions[0]);
}

// Split long windows across CTAs. Each warp streams one subset of rows
// and accumulates a local online softmax in registers.
constexpr int AttentionTile = 128;
constexpr int MaxSplits = (W + Block + AttentionTile - 1) / AttentionTile;

template <int LocalQHeads, int LocalKVHeads>
__global__ void attend_split(const Bf* query, const Bf* ring_k, const Bf* ring_v,
                             const Bf* block_k, const Bf* block_v, int position,
                             float* partial, const int* device_position = nullptr) {
  constexpr int LocalKV = LocalKVHeads * D;
  if (device_position) position = *device_position;
  constexpr int Warps = Threads / 32;
  __shared__ float numerators[Warps][D];
  __shared__ float peaks[Warps], sums[Warps];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int query_idx = blockIdx.x;
  const int kvhead = (query_idx % LocalQHeads) / (LocalQHeads / LocalKVHeads);
  const int begin = max(0, position + query_idx / LocalQHeads - W + 1);
  const int first = begin + blockIdx.y * AttentionTile;
  const int end = min(first + AttentionTile, position + Block);
  float peak = -CUDART_INF_F, sum = 0.f;
  float acc[4] = {};
  const auto* q = reinterpret_cast<const __nv_bfloat162*>(query + query_idx * D);
  const float2 q0 = __bfloat1622float2(q[2 * lane]);
  const float2 q1 = __bfloat1622float2(q[2 * lane + 1]);
  for (int pos = first + warp; pos < end; pos += Warps) {
    const int offset = (pos < position ? pos % W : pos - position) * LocalKV + kvhead * D;
    const auto* k = reinterpret_cast<const __nv_bfloat162*>(
        (pos < position ? ring_k : block_k) + offset);
    const auto* v = reinterpret_cast<const __nv_bfloat162*>(
        (pos < position ? ring_v : block_v) + offset);
    const float2 k0 = __bfloat1622float2(k[2 * lane]);
    const float2 k1 = __bfloat1622float2(k[2 * lane + 1]);
    float dot = q0.x * k0.x + q0.y * k0.y + q1.x * k1.x + q1.y * k1.y;
    for (int delta = 16; delta; delta /= 2) dot += __shfl_down_sync(0xffffffff, dot, delta);
    const float score = __shfl_sync(0xffffffff, dot, 0) * rsqrtf(float(D));
    const float next = fmaxf(peak, score);
    const float carry = __expf(peak - next);
    const float weight = __expf(score - next);
    sum = sum * carry + weight;
    peak = next;
    const float2 v0 = __bfloat1622float2(v[2 * lane]);
    const float2 v1 = __bfloat1622float2(v[2 * lane + 1]);
    acc[0] = acc[0] * carry + weight * v0.x;
    acc[1] = acc[1] * carry + weight * v0.y;
    acc[2] = acc[2] * carry + weight * v1.x;
    acc[3] = acc[3] * carry + weight * v1.y;
  }
  if (lane == 0) {
    peaks[warp] = peak;
    sums[warp] = sum;
  }
  for (int i = 0; i < 4; ++i) numerators[warp][lane * 4 + i] = acc[i];
  __syncthreads();
  peak = -CUDART_INF_F;
  for (int i = 0; i < Warps; ++i) peak = fmaxf(peak, peaks[i]);
  sum = 0.f;
  float numerator = 0.f;
  for (int i = 0; i < Warps; ++i) {
    const float factor = sums[i] > 0.f ? __expf(peaks[i] - peak) : 0.f;
    sum += factor * sums[i];
    numerator += factor * numerators[i][threadIdx.x];
  }
  partial += (query_idx * MaxSplits + blockIdx.y) * (D + 2);
  partial[threadIdx.x] = numerator;
  if (threadIdx.x == 0) {
    partial[D] = peak;
    partial[D + 1] = sum;
  }
}

// Reuse each 128-bit K/V load across all 32 queries in one KV group.
constexpr int GqaTile = 32, QPerKV = QHeads / KVHeads, GqaRows = Block * QPerKV;
constexpr int GqaSplits = (W + Block + GqaTile - 1) / GqaTile;

template <int LocalQHeads, int LocalKVHeads>
__global__ void gqa_attention(const Bf* query, const Bf* ring_k, const Bf* ring_v,
                              const Bf* block_k, const Bf* block_v, int position,
                              float* partial, const int* device_position) {
  constexpr int LocalKV = LocalKVHeads * D;
  static_assert(LocalQHeads / LocalKVHeads == QPerKV);
  if (device_position) position = *device_position;
  using namespace nvcuda::wmma;
  __shared__ __align__(32) Bf qs[GqaRows][D], ks[GqaTile][D], vs[GqaTile][D];
  __shared__ __align__(32) float scores[GqaRows][GqaTile];
  __shared__ __align__(32) Bf probs[GqaRows][GqaTile];
  __shared__ __align__(32) float scratch[4][16][16];
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int head = blockIdx.x;
  const int begin = max(0, position - W + 1);
  const int first = begin + blockIdx.y * GqaTile;
  const int end = min(first + GqaTile, position + Block);
  if (first >= end) return;
  for (int i = tid; i < GqaTile * D / 8; i += Threads) {
    const int row = i / (D / 8), col = i % (D / 8);
    const int pos = first + row;
    const int qidx = (row / QPerKV) * LocalQHeads + head * QPerKV + row % QPerKV;
    reinterpret_cast<uint4*>(qs[row])[col] = reinterpret_cast<const uint4*>(query + qidx * D)[col];
    uint4 k = make_uint4(0, 0, 0, 0), v = k;
    if (pos < end) {
      const int off = (pos < position ? pos % W : pos - position) * LocalKV + head * D;
      k = reinterpret_cast<const uint4*>((pos < position ? ring_k : block_k) + off)[col];
      v = reinterpret_cast<const uint4*>((pos < position ? ring_v : block_v) + off)[col];
    }
    reinterpret_cast<uint4*>(ks[row])[col] = k;
    reinterpret_cast<uint4*>(vs[row])[col] = v;
  }
  __syncthreads();
  fragment<matrix_a, 16, 16, 16, Bf, row_major> a;
  fragment<matrix_b, 16, 16, 16, Bf, col_major> b;
  fragment<accumulator, 16, 16, 16, float> c;
  fill_fragment(c, 0.f);
  for (int d = 0; d < D; d += 16) {
    load_matrix_sync(a, &qs[16 * (warp / 2)][d], D);
    load_matrix_sync(b, &ks[16 * (warp % 2)][d], D);
    mma_sync(c, a, b, c);
  }
  store_matrix_sync(&scores[16 * (warp / 2)][16 * (warp % 2)], c, GqaTile, mem_row_major);
  __syncthreads();
  // Each warp reduces one query row at a time. The left edge advances
  // for each proposal token, while all proposal K/V remain visible.
  for (int qr = warp; qr < GqaRows; qr += 4) {
    const int qrow = qr / QPerKV, qidx = qrow * LocalQHeads + head * QPerKV + qr % QPerKV;
    const int pos = first + lane;
    const bool valid = pos < end && pos >= max(0, position + qrow - W + 1);
    const float score = valid ? scores[qr][lane] * rsqrtf(float(D)) : -CUDART_INF_F;
    float peak = score;
    for (int i = 16; i; i /= 2) peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, i));
    const float value = valid ? __expf(score - peak) : 0.f;
    probs[qr][lane] = __float2bfloat16(value);
    float sum = value;
    for (int i = 16; i; i /= 2) sum += __shfl_xor_sync(0xffffffff, sum, i);
    if (lane == 0) {
      float* dest = partial + (qidx * GqaSplits + blockIdx.y) * (D + 2);
      dest[D] = peak;
      dest[D + 1] = sum;
    }
  }
  __syncthreads();
  // Multiply the BF16 exponentials relative to the tile maximum by the shared
  // values. The cross-tile merge uses the FP32 peaks and denominators.
  fragment<matrix_b, 16, 16, 16, Bf, row_major> v;
  for (int d = (warp % 2) * 64; d < (warp % 2 + 1) * 64; d += 16) {
    fill_fragment(c, 0.f);
    for (int t = 0; t < GqaTile; t += 16) {
      load_matrix_sync(a, &probs[(warp / 2) * 16][t], GqaTile);
      load_matrix_sync(v, &vs[t][d], D);
      mma_sync(c, a, v, c);
    }
    store_matrix_sync(&scratch[warp][0][0], c, 16, mem_row_major);
    __syncwarp();
    for (int i = lane; i < 256; i += 32) {
      const int qr = (warp / 2) * 16 + i / 16;
      const int qidx = (qr / QPerKV) * LocalQHeads + head * QPerKV + qr % QPerKV;
      partial[(qidx * GqaSplits + blockIdx.y) * (D + 2) + d + i % 16] = scratch[warp][i / 16][i % 16];
    }
    __syncwarp();
  }
}

template <int Slots>
__global__ void merge_attention(const float* partial, int splits, Bf* out,
                                const int* device_position) {
  if (device_position)
    splits = (min(*device_position, W - 1) + Block + GqaTile - 1) / GqaTile;
  partial += blockIdx.x * Slots * (D + 2);
  __shared__ float factors[Slots], denominator;
  if (threadIdx.x == 0) {
    float peak = -CUDART_INF_F;
    for (int i = 0; i < splits; ++i) peak = fmaxf(peak, partial[i * (D + 2) + D]);
    float sum = 0.f;
    for (int i = 0; i < splits; ++i) {
      const float* tile = partial + i * (D + 2);
      factors[i] = __expf(tile[D] - peak);
      sum += factors[i] * tile[D + 1];
    }
    denominator = sum;
  }
  __syncthreads();
  float numerator = 0.f;
  for (int i = 0; i < splits; ++i)
    numerator += factors[i] * partial[i * (D + 2) + threadIdx.x];
  out[blockIdx.x * D + threadIdx.x] = __float2bfloat16(numerator / denominator);
}

__global__ void input_ids(int64_t* ids, int* anchor_out, int anchor,
                          int* position_out, int position) {
  ids[threadIdx.x] = threadIdx.x ? DFlash2::mask_token : anchor;
  if (threadIdx.x == 0) {
    *anchor_out = anchor;
    if (position_out) *position_out = position;
  }
}

__global__ void best16(const float* values, const int* ids, float* out_values, int* out_ids) {
  constexpr int Pool = kernels::draft::CANDIDATES;
  __shared__ float scores[Pool];
  __shared__ head::Candidate best[Threads];
  const int tid = threadIdx.x;
  values += blockIdx.x * Pool;
  ids += blockIdx.x * Pool;
  for (int i = tid; i < Pool; i += Threads) scores[i] = values[i];
  __syncthreads();
  for (int rank = 0; rank < 16; ++rank) {
    head::Candidate local{-CUDART_INF_F, INT_MAX};
    for (int i = tid; i < Pool; i += Threads)
      if (scores[i] > local.val || (scores[i] == local.val && ids[i] < local.global_id))
        local = {scores[i], ids[i]};
    best[tid] = local;
    __syncthreads();
    for (int stride = Threads / 2; stride; stride /= 2) {
      if (tid < stride) {
        const auto other = best[tid + stride];
        if (other.val > best[tid].val ||
            (other.val == best[tid].val && other.global_id < best[tid].global_id))
          best[tid] = other;
      }
      __syncthreads();
    }
    const auto winner = best[0];
    __syncthreads();
    if (tid == 0) {
      out_values[blockIdx.x * 16 + rank] = winner.val;
      out_ids[blockIdx.x * 16 + rank] = winner.global_id;
    }
    for (int i = tid; i < Pool; i += Threads)
      if (ids[i] == winner.global_id) scores[i] = -CUDART_INF_F;
    __syncthreads();
  }
}

enum class ShardMode { Replicated, RowSlice, ColSlice };

struct Linear {
  const Bf* bf16 = nullptr;
  head::Nvfp4Matrix packed;
  kernels::dense::Matrix fp8;
  int rows = 0, cols = 0;
};

struct Conv {
  const Bf* base;
  Linear projection;
};

struct DraftLayer {
  const Bf *pre, *post, *qnorm, *knorm;
  Conv attention, mlp;
  Linear q, k, v, o, gate, up, down;
  Bf *keys, *values;
};
}  // namespace

namespace {
struct DraftProfile {
  enum Stage { Commit, Embedding, Linear, Attention, Conv, Comm, Head, Selector, Count };
  struct Span {
    cudaEvent_t begin = nullptr, end = nullptr;
    Stage stage;
    ~Span() {
      if (begin) cudaEventDestroy(begin);
      if (end) cudaEventDestroy(end);
    }
  };
  cudaStream_t stream;
  bool enabled;
  bool capturing = false;
  bool pending_commit = false;
  std::array<Span, 64> spans;
  int used = 0, rounds = 0, commits = 0;
  int position = 0;
  std::array<double, Count> totals{};

  void begin(Stage stage) {
    if (!enabled) return;
    auto& span = spans.at(used);
    if (!span.begin) CUDA_CHECK(cudaEventCreate(&span.begin));
    if (!span.end) CUDA_CHECK(cudaEventCreate(&span.end));
    span.stage = stage;
    CUDA_CHECK(cudaEventRecordWithFlags(span.begin, stream, capturing ? cudaEventRecordExternal : 0));
  }

  void end() {
    if (enabled)
      CUDA_CHECK(cudaEventRecordWithFlags(spans.at(used++).end, stream, capturing ? cudaEventRecordExternal : 0));
  }

  void finish(int current_position) {
    if (!enabled || used == 0) return;
    if (pending_commit) ++commits;
    pending_commit = false;
    // Profiling is opt-in. This wait is excluded from the GPU stage totals.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i = 0; i < used; ++i) {
      float ms;
      CUDA_CHECK(cudaEventElapsedTime(&ms, spans[i].begin, spans[i].end));
      totals[spans[i].stage] += ms;
    }
    used = 0;
    position = current_position;
    ++rounds;
  }

  ~DraftProfile() { print(); }

  void print() const {
    if (!rounds) return;
    const char* names[Count] = {"commit", "embed/final_norm", "5L linear", "5L attn",
                                "5L conv/norm", "5L allreduce", "head/topk", "selector"};
    std::fprintf(stderr, "DFlash2 profile (%d rounds, final position %d)", rounds, position);
    double total = 0;
    for (int i = 0; i < Count; ++i) {
      const double mean = totals[i] / rounds;
      total += mean;
      std::fprintf(stderr, " | %s %.4f ms", names[i], mean);
    }
    std::fprintf(stderr, " | stages %.4f ms", total);
    if (commits)
      std::fprintf(stderr, " | commit actual %.4f ms (%d commits)", totals[Commit] / commits, commits);
    std::fprintf(stderr, "\n");
  }

  void reset() {
    print();
    used = rounds = commits = 0;
    pending_commit = false;
    totals.fill(0);
  }
};
}  // namespace

namespace {
__global__ void cumulative_confidence(float* confidence) {
  float product = 1.f;
  for (int i = 0; i < Proposals; ++i) {
    product *= confidence[i];
    confidence[i] = product;
  }
}
}  // namespace

struct DFlash2::Impl {
  cudaStream_t s;
  TpRuntime* tp = nullptr;
  int tp_size = 1;
  int tp_rank = 0;
  int q_heads = QHeads, kv_heads = KVHeads;
  int q_dim = Q, kv_dim = KV, i_dim = I;
  bool shard_fp32 = false;
  DraftProfile profile;
  int length = 0;
  int* device_position = nullptr;
  bool warmed = false;
  bool gqa = false;
  std::array<cudaGraphExec_t, MaxSplits + 1> graphs{};
  std::array<int, MaxSplits + 1> graph_profile_spans{};
  ~Impl() {
    for (auto graph : graphs)
      if (graph) cudaGraphExecDestroy(graph);
  }

  std::size_t ring_bytes() const { return std::size_t(W) * kv_dim * sizeof(Bf); }

  int attention_splits() const {
    if (length + Block <= AttentionTile) return 0;
    return (std::min(length, W - 1) + Block + AttentionTile - 1) / AttentionTile;
  }
  std::vector<DeviceBuffer> buffers;
  kernels::draft::Bf16Gemm gemm;
  std::array<DraftLayer, 5> layers;
  Linear fc, selector;
  Bf* padded_taps = nullptr;
  const Bf *hidden_norm = nullptr, *final_norm = nullptr, *pred = nullptr, *succ = nullptr;
  Bf *ctx = nullptr, *commit_k = nullptr, *commit_v = nullptr, *x = nullptr, *normed = nullptr;
  Bf *convolved = nullptr, *coeff = nullptr;
  Bf *q = nullptr, *key = nullptr, *value = nullptr, *block_key = nullptr, *attention = nullptr;
  Bf *projection = nullptr, *gate = nullptr, *up = nullptr, *projected = nullptr;
  float* attention_partial = nullptr;
  float* partial_f32 = nullptr;
  float *fp4_scratch = nullptr, *lattice = nullptr, *confidence = nullptr;
  int* tokens = nullptr;
  head::Nvfp4Matrix target_head;
  kernels::draft::DFlash2DraftHead draft_head;
  DFlash2Head* sharded_head = nullptr;
  const Bf* embed = nullptr;
  const float* embed_norm = nullptr;
  Bf* embeddings = nullptr;
  int64_t* ids = nullptr;
  float* embed_output = nullptr;
  std::filesystem::path dump_dir;
  const bool cumulative = std::getenv("INKLING_DFLASH2_CUMULATIVE_CONFIDENCE") != nullptr;
  bool dump_written = false;
  int dump_round = 0, proposal_round = 0;
  float *unary = nullptr, *pool_values = nullptr, *top_values = nullptr;
  int *anchor_dev = nullptr, *pool_ids = nullptr, *top_ids = nullptr;
  void* top_workspace = nullptr;

  template <typename T = Bf>
  T* allocate(std::size_t n) {
    buffers.emplace_back(n * sizeof(T));
    return static_cast<T*>(buffers.back().p);
  }

  Impl(const std::filesystem::path& path, cudaStream_t stream, bool nvfp4,
       TpRuntime* tp_rt, int shard_size, int shard_rank)
      : s(stream),
        tp(tp_rt),
        tp_size(tp_rt ? tp_rt->layout.tp_size : shard_size),
        tp_rank(tp_rt ? tp_rt->layout.tp_rank : shard_rank),
        q_heads(QHeads / tp_size),
        kv_heads(KVHeads / tp_size),
        q_dim(Q / tp_size),
        kv_dim(KV / tp_size),
        i_dim(I / tp_size),
        shard_fp32(tp_size == 2 && dflash2_shard_fp32()),
        profile{stream, std::getenv("INKLING_DFLASH2_PROFILE") != nullptr && tp_rank == 0} {
    if ((tp_size != 1 && tp_size != 2) || tp_rank < 0 || tp_rank >= tp_size)
      throw std::invalid_argument("DFlash2 supports tp_size 1 or 2 and tp_rank in [0, tp_size)");
    if (const auto* dir = std::getenv("INKLING_DFLASH2_DUMP")) {
      if (tp_size != 1)
        throw std::invalid_argument("INKLING_DFLASH2_DUMP requires unsharded DFlash2");
      dump_dir = dir;
      if (const auto* round = std::getenv("INKLING_DFLASH2_DUMP_ROUND")) {
        std::size_t parsed = 0;
        dump_round = std::stoi(round, &parsed);
        if (dump_round < 0 || round[parsed] != '\0')
          throw std::invalid_argument("INKLING_DFLASH2_DUMP_ROUND must be a nonnegative integer");
      }
      std::filesystem::create_directories(dump_dir);
    }
    const auto* precision_env = std::getenv("INKLING_DFLASH2_PRECISION");
    const std::string_view precision = precision_env ? precision_env : nvfp4 ? "" : "fp8";
    if (!precision.empty() && precision != "fp8" && precision != "hybrid")
      throw std::invalid_argument("INKLING_DFLASH2_PRECISION must be fp8 or hybrid");
    if (!precision.empty() && nvfp4)
      throw std::invalid_argument("Choose either INKLING_DFLASH2_PRECISION or --draft-nvfp4");
    const bool hybrid = precision == "hybrid";
    if (dump_dir.empty())
      device_position = allocate<int>(1);
    gqa = dflash2_gqa();
    SafetensorsModel model(path);
    auto upload = [&](const std::string& name, std::vector<std::int64_t> shape, DeviceBuffer& buffer) {
      const auto tensor = model.at(name);
      if (tensor->dtype != "BF16" || tensor->shape != shape)
        throw std::runtime_error("Unexpected DFlash2 tensor shape or dtype: " + name);
      buffer = DeviceBuffer(tensor->bytes);
      const auto bytes = model.read(name);
      buffer.copy_from_host(bytes.data(), bytes.size());
      return static_cast<Bf*>(buffer.p);
    };
    auto load = [&](const std::string& name, std::vector<std::int64_t> shape) {
      buffers.emplace_back();
      return upload(name, std::move(shape), buffers.back());
    };
    auto linear = [&](const std::string& name, int full_rows, int full_cols, bool pack = false,
                      ShardMode shard = ShardMode::Replicated) {
      if (tp_size == 1) shard = ShardMode::Replicated;
      const int rows = shard == ShardMode::RowSlice ? full_rows / tp_size : full_rows;
      const int cols = shard == ShardMode::ColSlice ? full_cols / tp_size : full_cols;
      Linear layer;
      layer.rows = rows;
      layer.cols = cols;
      const bool mlp = name.find(".mlp.") != std::string::npos;
      const bool use_fp8 = pack && !precision.empty() && !(hybrid && mlp);
      const bool use_nvfp4 = pack && !use_fp8 && (nvfp4 || hybrid);
      if (shard == ShardMode::Replicated) {
        if (use_fp8) {
          DeviceBuffer source;
          const auto* weights = upload(name, {full_rows, full_cols}, source);
          auto* codes = allocate<uint8_t>(std::size_t(rows) * cols);
          auto* scale = allocate<float>(rows);
          kernels::dense::fp8_rows(weights, rows, cols, codes, scale, s);
          layer.fp8 = {codes, scale, rows, cols};
          if (name.ends_with(".k_proj.weight") || name.ends_with(".v_proj.weight")) {
            auto* mirror = allocate(std::size_t(rows) * cols);
            kernels::dense::dequant_rows(codes, scale, rows, cols, mirror, s);
            layer.bf16 = mirror;
          }
          CUDA_CHECK(cudaStreamSynchronize(s));
        } else if (use_nvfp4) {
          DeviceBuffer source;
          const auto* weights = upload(name, {full_rows, full_cols}, source);
          const auto elements = std::size_t(rows) * cols;
          auto* codes = allocate<uint8_t>(elements / 2);
          auto* scales = allocate<uint8_t>(elements / 16);
          const float scale = head::nvfp4_outer_scale(weights, elements, s);
          head::nvfp4_pack_rows(weights, rows, cols, scale, codes, scales, s);
          CUDA_CHECK(cudaStreamSynchronize(s));
          layer.packed = {codes, scales, scale, rows, 0};
        } else {
          layer.bf16 = load(name, {full_rows, full_cols});
        }
      } else if (shard == ShardMode::RowSlice) {
        DeviceBuffer source;
        const auto* full_weights = upload(name, {full_rows, full_cols}, source);
        const auto* weights = full_weights + std::size_t(tp_rank) * rows * cols;
        if (use_fp8) {
          auto* codes = allocate<uint8_t>(std::size_t(rows) * cols);
          auto* scale = allocate<float>(rows);
          kernels::dense::fp8_rows(weights, rows, cols, codes, scale, s);
          layer.fp8 = {codes, scale, rows, cols};
          // Bulk context commit uses the same quantized K/V weights through the
          // BF16 GEMM path. Keep mirrors only for the two commit projections.
          if (name.ends_with(".k_proj.weight") || name.ends_with(".v_proj.weight")) {
            auto* mirror = allocate(std::size_t(rows) * cols);
            kernels::dense::dequant_rows(codes, scale, rows, cols, mirror, s);
            layer.bf16 = mirror;
          }
          CUDA_CHECK(cudaStreamSynchronize(s));
        } else if (use_nvfp4) {
          const auto full_elements = std::size_t(full_rows) * full_cols;
          const auto elements = std::size_t(rows) * cols;
          auto* codes = allocate<uint8_t>(elements / 2);
          auto* scales = allocate<uint8_t>(elements / 16);
          // Calibrate on the full tensor so the outer scale matches unsharded packing.
          const float scale = head::nvfp4_outer_scale(full_weights, full_elements, s);
          head::nvfp4_pack_rows(weights, rows, cols, scale, codes, scales, s);
          CUDA_CHECK(cudaStreamSynchronize(s));
          layer.packed = {codes, scales, scale, rows, 0};
        } else {
          auto* dst = allocate(std::size_t(rows) * cols);
          CUDA_CHECK(cudaMemcpyAsync(dst, weights, std::size_t(rows) * cols * sizeof(Bf),
                                     cudaMemcpyDeviceToDevice, s));
          CUDA_CHECK(cudaStreamSynchronize(s));
          layer.bf16 = dst;
        }
      } else {
        // ShardMode::ColSlice (row-parallel o_proj / down_proj): quantize the full row first so
        // per-row FP8 scales / NVFP4 outer scale and codes match unsharded DFlash2 bit-for-bit.
        DeviceBuffer source;
        const auto* full_weights = upload(name, {full_rows, full_cols}, source);
        if (use_fp8) {
          DeviceBuffer full_codes_buf(std::size_t(full_rows) * full_cols);
          auto* scale = allocate<float>(full_rows);
          kernels::dense::fp8_rows(full_weights, full_rows, full_cols, full_codes_buf.p, scale, s);
          auto* codes = allocate<uint8_t>(std::size_t(rows) * cols);
          const auto* src_codes = static_cast<const uint8_t*>(full_codes_buf.p) + std::size_t(tp_rank) * cols;
          CUDA_CHECK(cudaMemcpy2DAsync(codes, cols, src_codes, full_cols, cols, rows,
                                       cudaMemcpyDeviceToDevice, s));
          CUDA_CHECK(cudaStreamSynchronize(s));
          layer.fp8 = {codes, scale, rows, cols};
        } else if (use_nvfp4) {
          const auto full_elements = std::size_t(full_rows) * full_cols;
          const float scale = head::nvfp4_outer_scale(full_weights, full_elements, s);
          DeviceBuffer full_codes_buf(full_elements / 2);
          DeviceBuffer full_scales_buf(full_elements / 16);
          head::nvfp4_pack_rows(full_weights, full_rows, full_cols, scale,
                                static_cast<uint8_t*>(full_codes_buf.p),
                                static_cast<uint8_t*>(full_scales_buf.p), s);
          const auto elements = std::size_t(rows) * cols;
          auto* codes = allocate<uint8_t>(elements / 2);
          auto* scales = allocate<uint8_t>(elements / 16);
          const auto* src_codes = static_cast<const uint8_t*>(full_codes_buf.p) + std::size_t(tp_rank) * (cols / 2);
          const auto* src_scales = static_cast<const uint8_t*>(full_scales_buf.p) + std::size_t(tp_rank) * (cols / 16);
          CUDA_CHECK(cudaMemcpy2DAsync(codes, cols / 2, src_codes, full_cols / 2, cols / 2, rows,
                                       cudaMemcpyDeviceToDevice, s));
          CUDA_CHECK(cudaMemcpy2DAsync(scales, cols / 16, src_scales, full_cols / 16, cols / 16, rows,
                                       cudaMemcpyDeviceToDevice, s));
          CUDA_CHECK(cudaStreamSynchronize(s));
          layer.packed = {codes, scales, scale, rows, 0};
        } else {
          auto* dst = allocate(std::size_t(rows) * cols);
          const auto* src = full_weights + std::size_t(tp_rank) * cols;
          CUDA_CHECK(cudaMemcpy2DAsync(dst, cols * sizeof(Bf), src, full_cols * sizeof(Bf),
                                       cols * sizeof(Bf), rows, cudaMemcpyDeviceToDevice, s));
          CUDA_CHECK(cudaStreamSynchronize(s));
          layer.bf16 = dst;
        }
      }
      return layer;
    };
    fc = linear("fc.weight", H, 4 * H);
    if (!precision.empty()) {
      auto* codes = allocate<uint8_t>(std::size_t(H) * 4 * H);
      auto* scale = allocate<float>(H);
      kernels::dense::fp8_rows(fc.bf16, H, 4 * H, codes, scale, s);
      fc.fp8 = {codes, scale, H, 4 * H};
      padded_taps = allocate(Block * 4 * H);
    }
    hidden_norm = load("hidden_norm.weight", {H});
    final_norm = load("norm.weight", {H});
    if (tp_rank == 0) {
      selector = linear("candidate_selector.hidden_projection.weight", Rank, H);
      pred = load("candidate_selector.predecessor_codebook", {Vocabulary, Rank});
      succ = load("candidate_selector.successor_codebook", {Vocabulary, Rank});
    }
    for (std::size_t i = 0; i < layers.size(); ++i) {
      auto& l = layers[i];
      const std::string prefix = "layers." + std::to_string(i) + ".";
      auto conv = [&](const std::string& name) {
        return Conv{load(prefix + name + ".base_kernel", {2, 2, H}),
                    linear(prefix + name + ".kernel_projection.weight", Coefficients, H)};
      };
      l.pre = load(prefix + "input_layernorm.weight", {H});
      l.post = load(prefix + "post_attention_layernorm.weight", {H});
      l.qnorm = load(prefix + "self_attn.q_norm.weight", {D});
      l.knorm = load(prefix + "self_attn.k_norm.weight", {D});
      l.attention = conv("attention_conv");
      l.mlp = conv("mlp_conv");
      l.q = linear(prefix + "self_attn.q_proj.weight", Q, H, true, ShardMode::RowSlice);
      l.k = linear(prefix + "self_attn.k_proj.weight", KV, H, true, ShardMode::RowSlice);
      l.v = linear(prefix + "self_attn.v_proj.weight", KV, H, true, ShardMode::RowSlice);
      l.o = linear(prefix + "self_attn.o_proj.weight", H, Q, true, ShardMode::ColSlice);
      l.gate = linear(prefix + "mlp.gate_proj.weight", I, H, true, ShardMode::RowSlice);
      l.up = linear(prefix + "mlp.up_proj.weight", I, H, true, ShardMode::RowSlice);
      l.down = linear(prefix + "mlp.down_proj.weight", H, I, true, ShardMode::ColSlice);
      l.keys = allocate(W * kv_dim);
      l.values = allocate(W * kv_dim);
    }
    ctx = allocate(CommitBatch * H);
    commit_k = allocate(CommitBatch * kv_dim);
    commit_v = allocate(CommitBatch * kv_dim);
    x = allocate(Block * H);
    normed = allocate(Block * H);
    convolved = allocate(Block * H);
    coeff = allocate(Block * Coefficients);
    q = allocate(Block * q_dim);
    key = allocate(Block * kv_dim);
    value = allocate(Block * kv_dim);
    block_key = allocate(Block * kv_dim);
    attention = allocate(Block * q_dim);
    attention_partial = allocate<float>(Block * q_heads * (gqa ? GqaSplits : MaxSplits) * (D + 2));
    // projection holds q_proj ([Block, q_dim]) as well as o_proj/down_proj ([Block, H]).
    projection = allocate(Block * std::max(q_dim, H));
    gate = allocate(Block * i_dim);
    up = allocate(Block * i_dim);
    if (tp_size == 2) partial_f32 = allocate<float>(Block * H);
    fp4_scratch = (nvfp4 || hybrid) ? allocate<float>(Block * std::max(i_dim, H)) : nullptr;
    if (tp_rank == 0) {
      projected = allocate(Proposals * Rank);
      lattice = allocate<float>(Proposals * 16 * 16);
      tokens = allocate<int>(Proposals);
      confidence = allocate<float>(Proposals);
    }
  }

  template <typename T>
  void dump(const char* name, const T* values, std::size_t count, bool append = false) {
    std::vector<T> data(count);
    CUDA_CHECK(cudaMemcpyAsync(data.data(), values, count * sizeof(T), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    std::ofstream file(dump_dir / name, std::ios::binary | (append ? std::ios::app : std::ios::trunc));
    if (!file.write(reinterpret_cast<const char*>(data.data()), count * sizeof(T)))
      throw std::runtime_error("Could not write DFlash2 fixture");
  }

  void linear(const Bf* input, int n, const Linear& w, Bf* out) {
    if (w.fp8.w && n == Block) {
      kernels::dense::dflash2_fp8(input, w.fp8, out, s);
    } else if (w.bf16) {
      gemm.run(input, n, w.bf16, w.rows, w.cols, out, s);
    } else {
      for (int first = 0; first < n; first += Block) {
        const int rows = std::min(Block, n - first);
        head::project_nvfp4(input + first * w.cols, rows, w.cols, w.packed, fp4_scratch, s);
        const int elements = rows * w.rows;
        convert<<<(elements + Threads - 1) / Threads, Threads, 0, s>>>(
            fp4_scratch, out + first * w.rows, elements);
      }
    }
  }

  void linear_f32(const Bf* input, const Linear& w, float* out) {
    if (w.fp8.w) {
      kernels::dense::dflash2_fp8_f32(input, w.fp8, out, s);
    } else if (w.packed.codes) {
      head::project_nvfp4(input, Block, w.cols, w.packed, out, s);
    } else {
      gemm.run(input, Block, w.bf16, w.rows, w.cols, projection, s);
      widen<<<(Block * w.rows + Threads - 1) / Threads, Threads, 0, s>>>(projection, out, Block * w.rows);
    }
  }

  void commit(const Bf* taps, int n, int start, bool decode) {
    if (start != length || n < 0)
      throw std::invalid_argument("DFlash2 commit must append to the committed context");
    if (n == 0) return;
    if (!dump_dir.empty() && !dump_written) dump("taps.bf16", taps, std::size_t(n) * 4 * H, start != 0);
    profile.used = 0;
    profile.pending_commit = decode && profile.enabled;
    if (profile.pending_commit) profile.begin(DraftProfile::Commit);
    const bool decode_fp8 = padded_taps && decode && n <= Block;
    if (decode_fp8 && n < Block) {
      CUDA_CHECK(cudaMemcpyAsync(padded_taps, taps, n * 4 * H * sizeof(Bf), cudaMemcpyDeviceToDevice, s));
      CUDA_CHECK(cudaMemsetAsync(padded_taps + n * 4 * H, 0, (Block - n) * 4 * H * sizeof(Bf), s));
    }
    // Any earlier rows in an oversized commit are outside the resulting ring.
    for (int first = std::max(0, n - W); first < n; first += CommitBatch) {
      const int count = std::min(CommitBatch, n - first);
      const int pos = start + first;
      const int projected_rows = decode_fp8 ? Block : count;
      if (decode_fp8) {
        kernels::dense::dflash2_fp8(n == Block ? taps : padded_taps, fc.fp8, ctx, s);
      } else {
        gemm.run(taps + std::size_t(first) * 4 * H, count, fc.bf16, H, 4 * H, ctx, s);
      }
      normalize<<<projected_rows, Threads, 0, s>>>(ctx, hidden_norm, H, 1, -1, ctx, 0);
      for (const auto& l : layers) {
        linear(ctx, projected_rows, l.k, commit_k);
        linear(ctx, projected_rows, l.v, commit_v);
        normalize<<<count * kv_heads, Threads, 0, s>>>(
            commit_k, l.knorm, D, kv_heads, pos, l.keys, W);
        const int slot = pos % W;
        const int contiguous = std::min(count, W - slot);
        CUDA_CHECK(cudaMemcpyAsync(l.values + slot * kv_dim, commit_v, contiguous * kv_dim * sizeof(Bf),
                                   cudaMemcpyDeviceToDevice, s));
        if (contiguous < count)
          CUDA_CHECK(cudaMemcpyAsync(l.values, commit_v + contiguous * kv_dim,
                                     (count - contiguous) * kv_dim * sizeof(Bf), cudaMemcpyDeviceToDevice, s));
      }
    }
    CUDA_CHECK(cudaGetLastError());
    length += n;
    if (decode) profile.end();
  }

  void bind_embeddings(const Bf* weight, const float* norm) {
    if (embed || !weight || !norm)
      throw std::invalid_argument("DFlash2 requires target embeddings");
    embed = weight;
    embed_norm = norm;
    embeddings = allocate(Block * H);
    ids = allocate<int64_t>(Block);
    embed_output = allocate<float>(Block * H);
    anchor_dev = allocate<int>(1);
  }

  void prepare(const Bf* weight, const Conv& conv) {
    normalize<<<Block, Threads, 0, s>>>(x, weight, H, 1, -1, normed, 0);
    linear(normed, Block, conv.projection, coeff);
    convolution<<<dim3((H + Threads - 1) / Threads, Block), Threads, 0, s>>>(
        normed, coeff, conv.base, 0, convolved, false);
  }

  void residual(const Conv& conv) {
    convolution<<<dim3((H + Threads - 1) / Threads, Block), Threads, 0, s>>>(
        projection, coeff, conv.base, 1, x, true);
  }

  template <int LocalQHeads, int LocalKVHeads>
  void run_attention(const DraftLayer& l, int splits, const int* position) {
    if (!splits) {
      attend<LocalQHeads, LocalKVHeads><<<Block * LocalQHeads, Threads, 0, s>>>(
          q, l.keys, l.values, block_key, value, length, attention, position);
    } else if (gqa) {
      const int tiles = (std::min(length, W - 1) + Block + GqaTile - 1) / GqaTile;
      const int grid_tiles = position ? GqaSplits : tiles;
      gqa_attention<LocalQHeads, LocalKVHeads><<<dim3(LocalKVHeads, grid_tiles), Threads, 0, s>>>(
          q, l.keys, l.values, block_key, value, length, attention_partial, position);
      merge_attention<GqaSplits><<<Block * LocalQHeads, Threads, 0, s>>>(
          attention_partial, tiles, attention, position);
    } else {
      attend_split<LocalQHeads, LocalKVHeads><<<dim3(Block * LocalQHeads, splits), Threads, 0, s>>>(
          q, l.keys, l.values, block_key, value, length, attention_partial, position);
      merge_attention<MaxSplits><<<Block * LocalQHeads, Threads, 0, s>>>(
          attention_partial, splits, attention, nullptr);
    }
  }

  void layer_attn_partial(int idx, int splits, const int* position, bool to_f32) {
    const auto& l = layers[idx];
    profile.begin(DraftProfile::Conv);
    prepare(l.pre, l.attention);
    profile.end();
    profile.begin(DraftProfile::Linear);
    linear(convolved, Block, l.q, projection);
    normalize<<<Block * q_heads, Threads, 0, s>>>(projection, l.qnorm, D, q_heads, length, q, 0, position);
    linear(convolved, Block, l.k, key);
    normalize<<<Block * kv_heads, Threads, 0, s>>>(key, l.knorm, D, kv_heads, length, block_key, 0, position);
    linear(convolved, Block, l.v, value);
    profile.end();
    profile.begin(DraftProfile::Attention);
    if (tp_size == 2) run_attention<QHeads / 2, KVHeads / 2>(l, splits, position);
    else run_attention<QHeads, KVHeads>(l, splits, position);
    profile.end();
    profile.begin(DraftProfile::Linear);
    if (to_f32) linear_f32(attention, l.o, partial_f32);
    else linear(attention, Block, l.o, projection);
    profile.end();
  }

  void layer_mlp_partial(int idx, bool to_f32) {
    const auto& l = layers[idx];
    profile.begin(DraftProfile::Conv);
    residual(l.attention);
    prepare(l.post, l.mlp);
    profile.end();
    profile.begin(DraftProfile::Linear);
    linear(convolved, Block, l.gate, gate);
    linear(convolved, Block, l.up, up);
    swiglu<<<Block * i_dim / Threads, Threads, 0, s>>>(gate, up, gate);
    if (to_f32) linear_f32(gate, l.down, partial_f32);
    else linear(gate, Block, l.down, projection);
    profile.end();
  }

  void layer_mlp_epilogue(int idx) {
    profile.begin(DraftProfile::Conv);
    residual(layers[idx].mlp);
    profile.end();
  }

  void reduce_partial(unsigned op, bool to_f32) {
    if (tp) {
      profile.begin(DraftProfile::Comm);
      if (to_f32) {
        tp->allreduce_f32(partial_f32, Block * H, TpSchedule::Proposal, op, s);
        convert<<<(Block * H + Threads - 1) / Threads, Threads, 0, s>>>(partial_f32, projection, Block * H);
      } else {
        tp->allreduce_bf16(projection, Block * H, TpSchedule::Proposal, op, s);
      }
      profile.end();
    } else if (to_f32) {
      convert<<<(Block * H + Threads - 1) / Threads, Threads, 0, s>>>(partial_f32, projection, Block * H);
    }
  }

  void project_head(const Bf* hidden) {
    const auto matrix = draft_head.matrix();
    head::project_nvfp4(hidden, Proposals, H, matrix, unary, s);
    kernels::draft::topk_candidates(unary, Proposals, matrix.rows, draft_head.token_ids(),
                                    pool_values, pool_ids, top_workspace, s);
    best16<<<Proposals, Threads, 0, s>>>(pool_values, pool_ids, top_values, top_ids);
  }

  const Bf* forward(const Bf* embeds, const int* position = nullptr) {
    const int splits = attention_splits();
    const bool to_f32 = tp_size == 2 && shard_fp32;
    CUDA_CHECK(cudaMemcpyAsync(x, embeds, Block * H * sizeof(Bf), cudaMemcpyDeviceToDevice, s));
    for (int i = 0; i < int(layers.size()); ++i) {
      layer_attn_partial(i, splits, position, to_f32);
      if (tp_size == 2) reduce_partial(1u + 2u * i, to_f32);
      layer_mlp_partial(i, to_f32);
      if (tp_size == 2) reduce_partial(2u + 2u * i, to_f32);
      layer_mlp_epilogue(i);

    }
    profile.begin(DraftProfile::Embedding);
    normalize<<<Block, Threads, 0, s>>>(x, final_norm, H, 1, -1, normed, 0);
    CUDA_CHECK(cudaGetLastError());
    profile.end();
    return normed + H;
  }
};

namespace {
constexpr int HeadCandidates = Proposals * 16;

__global__ void pack_head_candidates(const float* values, const int* ids, int offset, head::Candidate* out) {
  const int i = threadIdx.x;
  out[i] = {values[i], ids[i] + offset};
}

__global__ void merge_head_candidates(const head::Candidate* gathered, int* ids, float* values) {
  const int row = threadIdx.x;
  const auto* first = gathered + row * 16;
  const auto* second = first + HeadCandidates;
  int a = 0, b = 0;
  for (int i = 0; i < 16; ++i) {
    const auto x = first[a], y = second[b];
    const bool pick_first = x.val > y.val || (x.val == y.val && x.global_id < y.global_id);
    const auto candidate = pick_first ? first[a++] : second[b++];
    ids[row * 16 + i] = candidate.global_id;
    values[row * 16 + i] = candidate.val;
  }
}
}  // namespace

struct DFlash2Head::Impl {
  TpRuntime& tp;
  kernels::draft::DFlash2DraftHead draft_head;
  cudaStream_t stream;
  DeviceBuffer hidden{Proposals * H * sizeof(Bf)};
  DeviceBuffer logits{std::size_t(Proposals) * draft_head.matrix().rows * sizeof(float)};
  DeviceBuffer pool_values{Proposals * kernels::draft::CANDIDATES * sizeof(float)};
  DeviceBuffer pool_ids{Proposals * kernels::draft::CANDIDATES * sizeof(int)};
  DeviceBuffer top_values{HeadCandidates * sizeof(float)}, top_ids{HeadCandidates * sizeof(int)};
  DeviceBuffer local{HeadCandidates * sizeof(head::Candidate)}, gathered{2 * local.bytes};
  DeviceBuffer workspace{kernels::draft::topk_workspace_bytes(Proposals, draft_head.matrix().rows)};

  Impl(TpRuntime& tp_in, head::Nvfp4Matrix matrix, cudaStream_t s) : tp(tp_in), draft_head(matrix, s), stream(s) {}
};

DFlash2Head::DFlash2Head(TpRuntime& tp, head::Nvfp4Matrix matrix, cudaStream_t stream) {
  if (!matrix.codes || !matrix.scales)
    throw std::invalid_argument("DFlash2 sharded head requires --head-nvfp4");
  impl_ = std::make_unique<Impl>(tp, matrix, stream);
  CUDA_CHECK(cudaMemsetAsync(impl_->workspace.p, 0, impl_->workspace.bytes, stream));
}
DFlash2Head::~DFlash2Head() = default;

void DFlash2Head::run(const Bf* hidden, bool replicated, bool run_merge) {
  auto& d = *impl_;
  auto* input = const_cast<Bf*>(hidden);
  if (!replicated) {
    if (!input) input = static_cast<Bf*>(d.hidden.p);
    d.tp.exchange_draft_head(input, d.hidden.bytes, nullptr, d.stream);
  } else if (!input) {
    throw std::invalid_argument("DFlash2Head::run(replicated=true) requires hidden");
  }
  auto* logits = static_cast<float*>(d.logits.p);
  auto* pool_values = static_cast<float*>(d.pool_values.p);
  auto* pool_ids = static_cast<int*>(d.pool_ids.p);
  auto* values = static_cast<float*>(d.top_values.p);
  auto* ids = static_cast<int*>(d.top_ids.p);
  const auto matrix = d.draft_head.matrix();
  head::project_nvfp4(input, Proposals, H, matrix, logits, d.stream);
  kernels::draft::topk_candidates(logits, Proposals, matrix.rows, d.draft_head.token_ids(),
                                   pool_values, pool_ids, d.workspace.p, d.stream);
  best16<<<Proposals, Threads, 0, d.stream>>>(pool_values, pool_ids, values, ids);
  // A mapped shortlist already supplies global IDs and has v_begin == 0.
  pack_head_candidates<<<1, HeadCandidates, 0, d.stream>>>(
      values, ids, matrix.v_begin, static_cast<head::Candidate*>(d.local.p));
  d.tp.exchange_draft_head(d.local.p, d.local.bytes, d.gathered.p, d.stream);
  if (replicated ? run_merge : (hidden != nullptr))
    merge_head_candidates<<<1, Proposals, 0, d.stream>>>(
        static_cast<const head::Candidate*>(d.gathered.p), ids, values);
  CUDA_CHECK(cudaGetLastError());
}

const int* DFlash2Head::ids() const { return static_cast<const int*>(impl_->top_ids.p); }
const float* DFlash2Head::values() const { return static_cast<const float*>(impl_->top_values.p); }

DFlash2::DFlash2(const std::filesystem::path& path, cudaStream_t stream, bool nvfp4,
                 TpRuntime* tp, int tp_size, int tp_rank)
    : impl_(std::make_unique<Impl>(path, stream, nvfp4, tp, tp_size, tp_rank)) {}
DFlash2::~DFlash2() = default;

bool DFlash2::is_sharded() const { return impl_->tp != nullptr; }
int DFlash2::tp_size() const { return impl_->tp_size; }
int DFlash2::tp_rank() const { return impl_->tp_rank; }

void DFlash2::bind_head(head::Nvfp4Matrix matrix, const Bf* embed, const float* embed_norm) {
  auto& d = *impl_;
  if (matrix.v_begin || matrix.rows < kernels::draft::CANDIDATES ||
      matrix.rows > Vocabulary || !matrix.codes || !matrix.scales)
    throw std::invalid_argument("DFlash2 requires the complete NVFP4 target head");
  d.bind_embeddings(embed, embed_norm);
  d.target_head = matrix;
  // Oracle dumps use full-vocabulary tensors with token IDs as row indices.
  d.draft_head = kernels::draft::DFlash2DraftHead(matrix, d.s, d.dump_dir.empty());
  const int rows = d.draft_head.matrix().rows;
  d.unary = d.allocate<float>(Proposals * rows);
  d.pool_values = d.allocate<float>(Proposals * kernels::draft::CANDIDATES);
  d.pool_ids = d.allocate<int>(Proposals * kernels::draft::CANDIDATES);
  d.top_values = d.allocate<float>(Proposals * 16);
  d.top_ids = d.allocate<int>(Proposals * 16);
  const auto bytes = kernels::draft::topk_workspace_bytes(Proposals, rows);
  d.top_workspace = d.allocate<std::byte>(bytes);
  CUDA_CHECK(cudaMemsetAsync(d.top_workspace, 0, bytes, d.s));
}

void DFlash2::bind_sharded_head(DFlash2Head& head, const Bf* embed, const float* embed_norm) {
  auto& d = *impl_;
  if (!d.dump_dir.empty())
    throw std::invalid_argument("DFlash2 oracle dump requires the full head");
  d.sharded_head = &head;
  d.bind_embeddings(embed, embed_norm);
}

void DFlash2::bind_backbone_only(const Bf* embed, const float* embed_norm) {
  impl_->bind_embeddings(embed, embed_norm);
}

void DFlash2::warmup() {
  auto& d = *impl_;
  if (!d.device_position) return;  // Proposal dumps use the uncaptured path.

  DeviceBuffer taps(std::size_t(CommitBatch) * 4 * H * sizeof(Bf));
  CUDA_CHECK(cudaMemsetAsync(taps.p, 0, taps.bytes, d.s));
  d.commit(static_cast<const Bf*>(taps.p), 1, 0, false);
  propose(0, true);  // Initialize the kernels before capturing.
  d.profile.used = 0;
  propose(0, true);

  // Capture one graph for each split count. GQA has a single split graph.
  for (int length = AttentionTile - Block + 1; length < W; length += AttentionTile) {
    d.commit(static_cast<const Bf*>(taps.p), length - d.length, d.length, false);
    propose(0, true);
    if (d.gqa) break;
  }
  // When sharded (d.tp != nullptr), sharded_head runs inside every captured graph in lockstep.
  // When unsharded, rank 1 performs a single startup head exchange to match this call.
  if (d.sharded_head && !d.tp) propose_sharded_head(d.normed + H, false);
  CUDA_CHECK(cudaStreamSynchronize(d.s));
  reset();
}

const int* DFlash2::propose(int anchor) {
  return propose(anchor, false);
}

const int* DFlash2::propose(int anchor, bool warming_up) {
  auto& d = *impl_;
  const int vocab_limit = d.target_head.codes ? d.target_head.rows : Vocabulary;
  if (!d.embed || anchor < 0 || anchor >= vocab_limit)
    throw std::invalid_argument("Invalid DFlash2 anchor or missing target head");
  // Reserve a consistent commit span for every graph, including the initial proposal.
  if (d.profile.enabled && d.profile.used == 0) {
    d.profile.begin(DraftProfile::Commit);
    d.profile.end();
  }
  input_ids<<<1, Block, 0, d.s>>>(d.ids, d.anchor_dev, anchor, d.device_position, d.length);
  const bool in_graph_sharded_head = d.sharded_head && d.tp != nullptr;
  auto launch = [&](const int* position) {
    d.profile.begin(DraftProfile::Embedding);
    kernels::dense::embed(d.ids, Block, d.embed, d.embed_norm, d.embed_output, false, d.s);
    convert<<<Block * H / Threads, Threads, 0, d.s>>>(d.embed_output, d.embeddings, Block * H);
    d.profile.end();
    const auto* hidden = d.forward(d.embeddings, position);
    if (in_graph_sharded_head) {
      // Replicated hidden state across both ranks: skip hidden broadcast and capture the sharded
      // head + candidate allgather (+ rank-0 selector) right inside the CUDA graph.
      propose_sharded_head(hidden, true);
      return;
    }
    if (d.sharded_head || !d.target_head.codes) return;
    d.profile.begin(DraftProfile::Head);
    d.project_head(hidden);
    d.profile.end();
    d.profile.begin(DraftProfile::Selector);
    select(d.top_ids, d.top_values, d.anchor_dev);
    d.profile.end();
  };
  if (d.device_position && d.warmed) {
    const int splits = d.attention_splits();
    const int graph_index = d.gqa && splits ? 1 : splits;
    auto& graph = d.graphs[graph_index];
    if (!graph) {
      d.profile.capturing = true;
      CUDA_CHECK(cudaStreamBeginCapture(d.s, cudaStreamCaptureModeThreadLocal));
      try {
        launch(d.device_position);
      } catch (...) {
        cudaGraph_t failed = nullptr;
        cudaStreamEndCapture(d.s, &failed);
        if (failed) cudaGraphDestroy(failed);
        throw;
      }
      d.profile.capturing = false;
      d.graph_profile_spans[graph_index] = d.profile.used;
      cudaGraph_t captured = nullptr;
      CUDA_CHECK(cudaStreamEndCapture(d.s, &captured));
      const auto status = cudaGraphInstantiate(&graph, captured, 0);
      cudaGraphDestroy(captured);
      CUDA_CHECK(status);
    }
    d.profile.used = d.graph_profile_spans[graph_index];
    CUDA_CHECK(cudaGraphLaunch(graph, d.s));
  } else {
    launch(nullptr);
    d.warmed = true;
  }
  const auto* hidden = d.normed + H;
  if (d.sharded_head && !in_graph_sharded_head && !warming_up) propose_sharded_head(hidden, false);
  if (!warming_up) d.profile.finish(d.length);
  const auto* tokens = d.tokens;
  if (!d.dump_dir.empty() && !d.dump_written && d.proposal_round++ == d.dump_round) {
    const auto elements = std::size_t(d.target_head.rows) * H;
    d.dump("head_codes.u8", d.target_head.codes, elements / 2);
    d.dump("head_scales.u8", d.target_head.scales, elements / 16);
    const float scale = d.target_head.outer_scale;
    std::ofstream scale_file(d.dump_dir / "head_scale.f32", std::ios::binary);
    if (!scale_file.write(reinterpret_cast<const char*>(&scale), sizeof(scale)))
      throw std::runtime_error("Could not write DFlash2 head scale");
    d.dump("embeddings.bf16", d.embeddings, Block * H);
    d.dump("hidden.bf16", hidden, Proposals * H);
    d.dump("unary.f32", d.unary, Proposals * d.target_head.rows);
    d.dump("top.i32", d.top_ids, Proposals * 16);
    d.dump("anchor.i32", d.anchor_dev, 1);
    d.dump("tokens.i32", tokens, Proposals);
    const int rows = std::min(d.length, W);
    const int first_slot = (d.length - rows) % W;
    const int contiguous = std::min(rows, W - first_slot);
    if (rows) {
      for (std::size_t i = 0; i < d.layers.size(); ++i) {
        const auto& layer = d.layers[i];
        for (const auto& [name, data] : {std::pair{"keys", layer.keys}, {"values", layer.values}}) {
          const auto file = std::string(name) + "-" + std::to_string(i) + ".bf16";
          d.dump(file.c_str(), data + first_slot * d.kv_dim, contiguous * d.kv_dim);
          if (contiguous < rows)
            d.dump(file.c_str(), data, (rows - contiguous) * d.kv_dim, true);
        }
      }
    }
    d.dump_written = true;
  }
  return tokens;
}

void DFlash2::propose_sharded_head(const Bf* hidden, bool replicated) {
  auto& d = *impl_;
  d.profile.begin(DraftProfile::Head);
  d.sharded_head->run(hidden, replicated, d.tp_rank == 0);
  d.profile.end();
  if (d.tp_rank == 0) {
    d.profile.begin(DraftProfile::Selector);
    select(d.sharded_head->ids(), d.sharded_head->values(), d.anchor_dev);
    d.profile.end();
  }
}

const float* DFlash2::confidence() const { return impl_->confidence; }
const int* DFlash2::tokens() const { return impl_->tokens; }

int batch_dflash2_max_ctx(int fallback, TpRuntime* tp, cudaStream_t stream) {
  int limit = fallback;
  const bool root = !tp || tp->layout.tp_rank == 0;
  if (root) {
    if (const char* env = std::getenv("INGINE_BATCH_DFLASH2_MAX_CTX"); env && *env) {
      const std::string_view value(env);
      const auto end = value.data() + value.size();
      const auto parsed = std::from_chars(value.data(), end, limit);
      if (parsed.ec != std::errc{} || parsed.ptr != end || limit < 0) limit = -1;
    }
  }
  if (tp) {
    // Batch subsets determine the collective schedule. Use rank 0's limit on both ranks.
    DeviceBuffer device(sizeof(limit));
    if (root)
      CUDA_CHECK(cudaMemcpyAsync(device.p, &limit, sizeof(limit), cudaMemcpyHostToDevice, stream));
    tp->broadcast_proposal(device.p, sizeof(limit), stream);
    CUDA_CHECK(cudaMemcpyAsync(&limit, device.p, sizeof(limit), cudaMemcpyDeviceToHost, stream));
    tp->wait(stream);
  }
  if (limit < 0) throw std::invalid_argument("INGINE_BATCH_DFLASH2_MAX_CTX must be a nonnegative integer");
  return limit;
}

bool dflash2_full_head() {
  const char* value = std::getenv("INKLING_DFLASH2_FULL_HEAD");
  return value && std::string_view(value) == "1";
}

bool dflash2_shard_backbone() {
  const char* value = std::getenv("INKLING_DFLASH2_SHARD_BACKBONE");
  return !value || std::string_view(value) != "0";
}

bool dflash2_gqa() {
  return std::getenv("INKLING_DFLASH2_GQA") != nullptr;
}

bool dflash2_shard_fp32() {
  const char* value = std::getenv("INKLING_DFLASH2_SHARD_FP32");
  return value && std::string_view(value) == "1";
}

void DFlash2::reset() {
  impl_->length = 0;
  impl_->proposal_round = 0;
  impl_->profile.reset();
}

int DFlash2::length() const { return impl_->length; }

std::size_t DFlash2::checkpoint_bytes() const {
  return 2 * impl_->layers.size() * impl_->ring_bytes();
}

void DFlash2::save_checkpoint(void* dst) {
  auto& d = *impl_;
  auto* buffer = static_cast<std::byte*>(dst);
  const std::size_t bytes = d.ring_bytes();
  for (const auto& layer : d.layers) {
    CUDA_CHECK(cudaMemcpyAsync(buffer, layer.keys, bytes, cudaMemcpyDeviceToDevice, d.s));
    CUDA_CHECK(cudaMemcpyAsync(buffer + bytes, layer.values, bytes, cudaMemcpyDeviceToDevice, d.s));
    buffer += 2 * bytes;
  }
  checkpoint_length_ = d.length;
}

void DFlash2::adopt_checkpoint(int len) {
  if (len < 0) throw std::invalid_argument("DFlash2 checkpoint length must be nonnegative");
  checkpoint_length_ = len;
}

void DFlash2::restore_checkpoint(const void* src, int len) {
  if (len < 0 || len != checkpoint_length_)
    throw std::invalid_argument("DFlash2 checkpoint length does not match the last save");
  auto& d = *impl_;
  auto* buffer = static_cast<const std::byte*>(src);
  const std::size_t bytes = d.ring_bytes();
  for (const auto& layer : d.layers) {
    CUDA_CHECK(cudaMemcpyAsync(layer.keys, buffer, bytes, cudaMemcpyDeviceToDevice, d.s));
    CUDA_CHECK(cudaMemcpyAsync(layer.values, buffer + bytes, bytes, cudaMemcpyDeviceToDevice, d.s));
    buffer += 2 * bytes;
  }
  d.length = len;
  d.proposal_round = 0;
  d.profile.used = 0;
  d.profile.pending_commit = false;
}

void DFlash2::commit(const Bf* taps, int n, int start, bool decode) {
  impl_->commit(taps, n, start, decode);
}

const Bf* DFlash2::forward(const Bf* embeddings) { return impl_->forward(embeddings); }

const int* DFlash2::select(const int* candidates, const float* unary, const int* anchor) {
  auto& d = *impl_;
  d.linear(d.normed + H, Proposals, d.selector, d.projected);
  kernels::draft::dflash2_greedy_selector(candidates, unary, d.projected, d.pred, d.succ,
                                          anchor, d.lattice, d.tokens, d.s, d.confidence);
  if (d.cumulative) cumulative_confidence<<<1, 1, 0, d.s>>>(d.confidence);
  return d.tokens;
}

const Bf* DFlash2::forward_pair(DFlash2& rank0, DFlash2& rank1, const Bf* embeddings,
                                bool fp32_reduce) {
  auto& d0 = *rank0.impl_;
  auto& d1 = *rank1.impl_;
  if (d0.tp_size != 2 || d0.tp_rank != 0 || d1.tp_size != 2 || d1.tp_rank != 1)
    throw std::invalid_argument("forward_pair requires rank0 (tp_size=2, rank=0) and rank1 (tp_size=2, rank=1)");
  if (d0.s != d1.s || d0.length != d1.length)
    throw std::invalid_argument("forward_pair requires matching stream and committed length");
  const cudaStream_t s = d0.s;
  const int splits = d0.attention_splits();
  CUDA_CHECK(cudaMemcpyAsync(d0.x, embeddings, Block * H * sizeof(Bf), cudaMemcpyDeviceToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(d1.x, embeddings, Block * H * sizeof(Bf), cudaMemcpyDeviceToDevice, s));
  for (int i = 0; i < int(d0.layers.size()); ++i) {
    d0.layer_attn_partial(i, splits, nullptr, fp32_reduce);
    d1.layer_attn_partial(i, splits, nullptr, fp32_reduce);
    if (fp32_reduce)
      sum_f32_pair_to_bf16<<<(Block * H + Threads - 1) / Threads, Threads, 0, s>>>(
          d0.partial_f32, d1.partial_f32, d0.projection, d1.projection, Block * H);
    else
      sum_bf16_pair<<<(Block * H + Threads - 1) / Threads, Threads, 0, s>>>(
          d0.projection, d1.projection, d0.projection, d1.projection, Block * H);
    d0.layer_mlp_partial(i, fp32_reduce);
    d1.layer_mlp_partial(i, fp32_reduce);
    if (fp32_reduce)
      sum_f32_pair_to_bf16<<<(Block * H + Threads - 1) / Threads, Threads, 0, s>>>(
          d0.partial_f32, d1.partial_f32, d0.projection, d1.projection, Block * H);
    else
      sum_bf16_pair<<<(Block * H + Threads - 1) / Threads, Threads, 0, s>>>(
          d0.projection, d1.projection, d0.projection, d1.projection, Block * H);
    d0.layer_mlp_epilogue(i);
    d1.layer_mlp_epilogue(i);

  }
  normalize<<<Block, Threads, 0, s>>>(d0.x, d0.final_norm, H, 1, -1, d0.normed, 0);
  normalize<<<Block, Threads, 0, s>>>(d1.x, d1.final_norm, H, 1, -1, d1.normed, 0);
  CUDA_CHECK(cudaGetLastError());
  return d0.normed + H;
}

const int* DFlash2::propose_pair(DFlash2& rank0, DFlash2& rank1, int anchor, bool fp32_reduce) {
  auto& d0 = *rank0.impl_;
  if (!d0.embed || !d0.target_head.codes || anchor < 0 || anchor >= d0.target_head.rows)
    throw std::invalid_argument("propose_pair requires rank0 with full bound head and valid anchor");
  input_ids<<<1, Block, 0, d0.s>>>(d0.ids, d0.anchor_dev, anchor, nullptr, d0.length);
  kernels::dense::embed(d0.ids, Block, d0.embed, d0.embed_norm, d0.embed_output, false, d0.s);
  convert<<<Block * H / Threads, Threads, 0, d0.s>>>(d0.embed_output, d0.embeddings, Block * H);
  const auto* hidden = forward_pair(rank0, rank1, d0.embeddings, fp32_reduce);
  d0.project_head(hidden);
  return rank0.select(d0.top_ids, d0.top_values, d0.anchor_dev);
}

}  // namespace ingine::model
