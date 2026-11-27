// Minimal GGUF (v2/v3) reader for TWinkling's GGUF expert mode: tensor index and raw byte ranges.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace ingine::model {

struct GgufTensorInfo {
  int type = 0;                 // ggml_type id
  std::vector<std::uint64_t> ne;  // innermost first
  std::uint64_t offset = 0;     // absolute file offset of the data
};

class GgufFile {
 public:
  explicit GgufFile(const std::filesystem::path& path);
  ~GgufFile();
  GgufFile(const GgufFile&) = delete;
  GgufFile& operator=(const GgufFile&) = delete;

  const GgufTensorInfo& at(const std::string& name) const;
  bool contains(const std::string& name) const { return tensors_.count(name) != 0; }
  const std::filesystem::path& path() const { return path_; }
  // Copies `bytes` from absolute file offset `offset` to device memory `dst` through pinned staging (stream-ordered,
  // synchronous on return).
  void read_to_device(std::uint64_t offset, std::uint64_t bytes, void* dst, cudaStream_t stream) const;

 private:
  std::filesystem::path path_;
  int fd_ = -1;
  std::uint64_t size_ = 0;
  std::unordered_map<std::string, GgufTensorInfo> tensors_;
};

}  // namespace ingine::model
