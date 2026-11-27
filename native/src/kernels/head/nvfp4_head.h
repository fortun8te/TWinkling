#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ingine::kernels::head {

struct Nvfp4Matrix {
  const uint8_t* codes = nullptr;   // [rows, cols / 2]
  const uint8_t* scales = nullptr;  // [rows, cols / 16] E4M3
  float outer_scale = 0;
  int rows = 0;
  int v_begin = 0;
};

struct Candidate {
  float val;
  int32_t global_id;
};

// Load-time calibration, shared by every vocabulary shard. Includes all checkpoint rows.
float nvfp4_outer_scale(const __nv_bfloat16* weights, std::size_t elements, cudaStream_t stream);
void nvfp4_pack_rows(const __nv_bfloat16* weights, int rows, float outer_scale,
                     uint8_t* codes, uint8_t* scales, cudaStream_t stream);

void nvfp4_pack_rows(const __nv_bfloat16* weights, int rows, int cols, float outer_scale,
                     uint8_t* codes, uint8_t* scales, cudaStream_t stream);

// BF16-rounded projections in FP32 scratch for DSpark's eight-row MLP.
void gemv_nvfp4(const __nv_bfloat16* x, int cols, Nvfp4Matrix w, float* out, cudaStream_t stream);

void project_nvfp4(const __nv_bfloat16* x, int M, int cols, Nvfp4Matrix w, float* out, cudaStream_t stream);

std::size_t nvfp4_candidate_workspace(int M, int rows);

// Supply at least logits [M, rows] or candidates [M]. Scratch is required only for candidates.
void head_nvfp4(const __nv_bfloat16* x, int M, Nvfp4Matrix w, float* logits, Candidate* candidates,
                Candidate* scratch, bool pdl, cudaStream_t stream);

// Certified head proxy. Writes both raw FP32 dots and a seed candidate per token.
void head_nvfp4_raw(const __nv_bfloat16* x, int M, Nvfp4Matrix w, float* dots, Candidate* candidates,
                    Candidate* scratch, cudaStream_t stream);

}  // namespace ingine::kernels::head
