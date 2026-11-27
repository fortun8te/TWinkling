#pragma once
// Single-GPU stand-in for Frankengine's tensor-parallel runtime: TWinkling runs DFlash2 on one RTX 5080,
// so every TpRuntime pointer is null and these members are never reached.
#include "model/tp_layout.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace ingine::model {

enum class TpSchedule : unsigned { Decode = 1, Prefill = 2, Request = 3, Proposal = 4, DraftHead = 5 };

class TpRuntime {
 public:
  const TpLayout layout{};
  [[noreturn]] static void unsupported() { throw std::logic_error("tensor parallelism is not built into TWinkling"); }
  void allreduce_bf16(__nv_bfloat16*, std::size_t, TpSchedule, unsigned, cudaStream_t) { unsupported(); }
  void allreduce_f32(float*, std::size_t, TpSchedule, unsigned, cudaStream_t) { unsupported(); }
  void broadcast_proposal(void*, std::size_t, cudaStream_t) { unsupported(); }
  void exchange_draft_head(void*, std::size_t, void*, cudaStream_t) { unsupported(); }
  void wait(cudaStream_t) { unsupported(); }
};

}  // namespace ingine::model
