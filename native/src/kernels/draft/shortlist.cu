#include "kernels/draft/shortlist.h"

#include "core/device.h"
#include "kernels/draft/draft.h"

#include <algorithm>
#include <cstdlib>
#include <string_view>
#include <vector>

namespace ingine::kernels::draft {
namespace {

constexpr int Vocab = 201024;
constexpr int Width = 2304;
constexpr int CodeBytes = Width / 2, ScaleBytes = Width / 16;
constexpr int CodeVectors = CodeBytes / sizeof(uint4), ScaleVectors = ScaleBytes / sizeof(std::uint16_t);
constexpr int Threads = 128;
constexpr std::size_t HotRows = 32768;
static_assert(CodeBytes % sizeof(uint4) == 0 && ScaleBytes % sizeof(std::uint16_t) == 0);
static_assert(CodeVectors == ScaleVectors && CodeVectors <= Threads);

__global__ void gather_nvfp4_rows(head::Nvfp4Matrix source, const std::int32_t* ids,
                                  uint4* codes, std::uint16_t* scales) {
  const int lane = threadIdx.x;
  if (lane >= CodeVectors) return;
  const int row = blockIdx.x;
  const int source_row = ids[row] - source.v_begin;
  const auto* src_codes = reinterpret_cast<const uint4*>(source.codes);
  const auto* src_scales = reinterpret_cast<const std::uint16_t*>(source.scales);
  codes[std::size_t(row) * CodeVectors + lane] = src_codes[std::size_t(source_row) * CodeVectors + lane];
  scales[std::size_t(row) * ScaleVectors + lane] = src_scales[std::size_t(source_row) * ScaleVectors + lane];
}

__global__ void remap_shortlist_ids(std::int32_t* ids, int count, const std::int32_t* token_ids) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) ids[i] = token_ids[ids[i]];
}

}  // namespace

DFlash2DraftHead::DFlash2DraftHead(head::Nvfp4Matrix source, cudaStream_t stream, bool use_shortlist)
    : matrix_(source) {
  if (!use_shortlist || !source.codes || !source.scales || source.rows < CANDIDATES ||
      source.v_begin < 0 || source.v_begin > Vocab - source.rows)
    return;
  // Opt-in only: static 32K shortlist drops all-73 multilingual/rare-token acceptance (2.93 -> 2.58 tok/step).
  std::size_t hot_rows = HotRows;
  const char* env = std::getenv("INGINE_DFLASH2_SHORTLIST");
  if (!env || !*env || std::string_view(env) == "0") return;
  if (std::string_view(env) != "1") {
    const long parsed = std::strtol(env, nullptr, 10);
    if (parsed <= 0) return;
    hot_rows = std::size_t(parsed);
  }

  constexpr std::uint32_t SpecialsBegin = 199990;  // always keep <|endoftext|>, <|end_message|>, <|message_model|>, etc.
  const auto ranks = draft_shortlist_ranks();
  if (hot_rows > ranks.size()) hot_rows = ranks.size();
  if (hot_rows < CANDIDATES) return;
  std::vector<std::uint32_t> ranked_ids(ranks.begin(), ranks.begin() + hot_rows);
  for (std::uint32_t id = SpecialsBegin; id < std::uint32_t(source.v_begin + source.rows); ++id)
    ranked_ids.push_back(id);
  // Compact row order must preserve the smallest-token-ID tie break in top-k.
  std::sort(ranked_ids.begin(), ranked_ids.end());
  ranked_ids.erase(std::unique(ranked_ids.begin(), ranked_ids.end()), ranked_ids.end());
  if (ranked_ids.back() >= Vocab) return;
  const auto first = std::lower_bound(ranked_ids.begin(), ranked_ids.end(), source.v_begin);
  const auto last = std::lower_bound(first, ranked_ids.end(), source.v_begin + source.rows);
  const std::vector<std::int32_t> ids(first, last);
  const auto rows = ids.size();
  if (rows < CANDIDATES || rows >= std::size_t(source.rows)) return;

  const auto map_bytes = rows * sizeof(std::int32_t);
  storage_ = DeviceBuffer(rows * (CodeBytes + ScaleBytes) + map_bytes);
  auto* codes = static_cast<std::uint8_t*>(storage_.p);
  auto* scales = codes + rows * CodeBytes;
  auto* map = reinterpret_cast<std::int32_t*>(scales + rows * ScaleBytes);
  CUDA_CHECK(cudaMemcpyAsync(map, ids.data(), map_bytes, cudaMemcpyHostToDevice, stream));
  gather_nvfp4_rows<<<int(rows), Threads, 0, stream>>>(
      source, map, reinterpret_cast<uint4*>(codes), reinterpret_cast<std::uint16_t*>(scales));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));  // The host ID array is used only during initialization.

  matrix_ = {codes, scales, source.outer_scale, int(rows), 0};
  token_ids_ = map;
}

void DFlash2DraftHead::remap_ids(std::int32_t* ids, int count, cudaStream_t stream) const {
  if (!token_ids_) return;
  remap_shortlist_ids<<<(count + Threads - 1) / Threads, Threads, 0, stream>>>(ids, count, token_ids_);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::draft
