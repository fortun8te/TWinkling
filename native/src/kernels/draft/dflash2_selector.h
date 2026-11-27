#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ingine::kernels::draft {

// Seven rows of 16 candidates, with unary logits before softcap. projected is
// [7,256], and the codebooks are [vocab,256]. anchor is a device token ID.
// Writes [7,16,16] edge scores and the seven tokens of the greedy local walk.
void dflash2_greedy_selector(const std::int32_t* ids, const float* unary,
                            const __nv_bfloat16* projected, const __nv_bfloat16* predecessors,
                            const __nv_bfloat16* successors, const std::int32_t* anchor,
                            float* lattice, std::int32_t* tokens, cudaStream_t stream,
                            float* confidence = nullptr);

}  // namespace ingine::kernels::draft
