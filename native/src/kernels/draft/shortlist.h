// The drafter's shortlist ranking (draft_shortlist.bin, built by native/tools/build_draft_shortlist.py and
// embedded at build time): token ids by descending corpus frequency, 65,536 entries.
#pragma once

#include "core/arena.h"
#include "kernels/head/nvfp4_head.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace ingine::kernels::draft {

std::span<const std::uint32_t> draft_shortlist_ranks();

// Owns a compact draft-only copy of a full or sharded NVFP4 vocabulary head.
// If disabled or the ranking cannot be used, matrix() returns the source head.
class DFlash2DraftHead {
 public:
  DFlash2DraftHead() = default;
  DFlash2DraftHead(head::Nvfp4Matrix source, cudaStream_t stream, bool use_shortlist = true);

  head::Nvfp4Matrix matrix() const { return matrix_; }
  const std::int32_t* token_ids() const { return token_ids_; }
  std::size_t device_bytes() const { return storage_.bytes; }

  // Convert compact row indices to global IDs after a top-k that does not take a map.
  void remap_ids(std::int32_t* ids, int count, cudaStream_t stream) const;

 private:
  head::Nvfp4Matrix matrix_;
  DeviceBuffer storage_;
  const std::int32_t* token_ids_ = nullptr;
};

}  // namespace ingine::kernels::draft
