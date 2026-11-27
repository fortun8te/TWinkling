#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "kernels/head/nvfp4_head.h"

#include <cstddef>
#include <filesystem>
#include <memory>

namespace ingine::model {

// In-order committed context. Proposal KV is held separately from the ring.
bool dflash2_full_head();
bool dflash2_shard_backbone();
bool dflash2_gqa();
bool dflash2_shard_fp32();

class DFlash2Head;
class TpRuntime;

int batch_dflash2_max_ctx(int fallback, TpRuntime* tp, cudaStream_t stream);

class DFlash2 {
 public:
  static constexpr int block = 8;
  static constexpr int mask_token = 200056;
  DFlash2(const std::filesystem::path& checkpoint, cudaStream_t stream, bool nvfp4,
          TpRuntime* tp = nullptr, int tp_size = 1, int tp_rank = 0);
  ~DFlash2();

  void bind_head(kernels::head::Nvfp4Matrix head, const __nv_bfloat16* embed,
                 const float* embed_norm);
  void bind_sharded_head(DFlash2Head& head, const __nv_bfloat16* embed, const float* embed_norm);
  void bind_backbone_only(const __nv_bfloat16* embed, const float* embed_norm);
  bool is_sharded() const;
  int tp_size() const;
  int tp_rank() const;
  void warmup();
  const int* propose(int anchor);
  const float* confidence() const;
  const int* tokens() const;  // the last proposal's 7 drafts (fixed device address)
  void reset();
  int length() const;
  // The caller supplies a device buffer of checkpoint_bytes() on this rank.
  std::size_t checkpoint_bytes() const;
  void save_checkpoint(void* dst);
  void adopt_checkpoint(int len);
  void restore_checkpoint(const void* src, int len);
  void commit(const __nv_bfloat16* taps, int n, int start, bool decode = false);
  const __nv_bfloat16* forward(const __nv_bfloat16* embeddings);
  const int* select(const int* candidates, const float* unary, const int* anchor);

  // Single-GPU two-rank simulation helpers for parity verification (rank0 and rank1 constructed with
  // tp == nullptr, tp_size == 2, tp_rank == 0 / 1 on the same stream).
  static const __nv_bfloat16* forward_pair(DFlash2& rank0, DFlash2& rank1,
                                           const __nv_bfloat16* embeddings,
                                           bool fp32_reduce = false);
  static const int* propose_pair(DFlash2& rank0, DFlash2& rank1, int anchor,
                                 bool fp32_reduce = false);

 private:
  const int* propose(int anchor, bool warming_up);
  void propose_sharded_head(const __nv_bfloat16* hidden, bool replicated);
  struct Impl;
  std::unique_ptr<Impl> impl_;
  int checkpoint_length_ = -1;
};

class DFlash2Head {
 public:
  DFlash2Head(TpRuntime& tp, kernels::head::Nvfp4Matrix matrix, cudaStream_t stream);
  ~DFlash2Head();
  void run(const __nv_bfloat16* hidden = nullptr, bool replicated = false, bool run_merge = true);
  const int* ids() const;
  const float* values() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ingine::model
