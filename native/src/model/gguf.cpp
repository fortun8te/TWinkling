#include "model/gguf.h"

#include "core/device.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ingine::model {
namespace {

struct Cursor {
  int fd;
  std::uint64_t pos = 0, size;
  void read(void* dst, std::size_t n) {
    std::size_t done = 0;
    while (done < n) {
      const ssize_t r = ::pread(fd, static_cast<char*>(dst) + done, n - done, off_t(pos + done));
      if (r <= 0) throw std::runtime_error("gguf: truncated header");
      done += std::size_t(r);
    }
    pos += n;
  }
  template <class T>
  T get() {
    T v;
    read(&v, sizeof v);
    return v;
  }
  std::string str() {
    const auto n = get<std::uint64_t>();
    if (n > (1u << 24)) throw std::runtime_error("gguf: string too long");
    std::string s(n, '\0');
    read(s.data(), n);
    return s;
  }
  void skip_value(std::uint32_t type) {
    static const std::size_t sizes[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    if (type == 8) {
      str();
    } else if (type == 9) {
      const auto et = get<std::uint32_t>();
      const auto n = get<std::uint64_t>();
      if (et == 8 || et == 9) {
        for (std::uint64_t i = 0; i < n; ++i) skip_value(et);
      } else {
        if (et > 12) throw std::runtime_error("gguf: bad array type");
        pos += n * sizes[et];
      }
    } else {
      if (type > 12) throw std::runtime_error("gguf: bad value type");
      pos += sizes[type];
    }
  }
};

}  // namespace

GgufFile::GgufFile(const std::filesystem::path& path) : path_(path) {
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) throw std::runtime_error("gguf: cannot open " + path.string());
  struct stat st {};
  ::fstat(fd_, &st);
  size_ = std::uint64_t(st.st_size);
  Cursor c{fd_, 0, size_};
  if (c.get<std::uint32_t>() != 0x46554747u) throw std::runtime_error("gguf: bad magic in " + path.string());
  const auto version = c.get<std::uint32_t>();
  if (version < 2) throw std::runtime_error("gguf: version 1 is not supported");
  const auto n_tensors = c.get<std::uint64_t>();
  const auto n_kv = c.get<std::uint64_t>();
  std::uint64_t alignment = 32;
  for (std::uint64_t i = 0; i < n_kv; ++i) {
    const std::string key = c.str();
    const auto type = c.get<std::uint32_t>();
    if (key == "general.alignment" && type == 4) alignment = c.get<std::uint32_t>();
    else c.skip_value(type);
  }
  std::vector<std::pair<std::string, GgufTensorInfo>> infos;
  infos.reserve(n_tensors);
  for (std::uint64_t i = 0; i < n_tensors; ++i) {
    GgufTensorInfo t;
    std::string name = c.str();
    const auto nd = c.get<std::uint32_t>();
    if (nd > 4) throw std::runtime_error("gguf: tensor rank > 4");
    t.ne.resize(nd);
    for (auto& d : t.ne) d = c.get<std::uint64_t>();
    t.type = int(c.get<std::uint32_t>());
    t.offset = c.get<std::uint64_t>();
    infos.emplace_back(std::move(name), std::move(t));
  }
  const std::uint64_t data = (c.pos + alignment - 1) / alignment * alignment;
  for (auto& [name, t] : infos) {
    t.offset += data;
    tensors_.emplace(name, t);
  }
}

GgufFile::~GgufFile() {
  if (fd_ >= 0) ::close(fd_);
}

const GgufTensorInfo& GgufFile::at(const std::string& name) const {
  const auto it = tensors_.find(name);
  if (it == tensors_.end()) throw std::runtime_error("gguf: no tensor " + name + " in " + path_.string());
  return it->second;
}

void GgufFile::read_to_device(std::uint64_t offset, std::uint64_t bytes, void* dst, cudaStream_t stream) const {
  if (offset + bytes > size_) throw std::runtime_error("gguf: range past the end of " + path_.string());
  constexpr std::size_t kChunk = std::size_t(64) << 20;
  void* staging[2] = {nullptr, nullptr};
  cudaEvent_t done[2];
  for (int i = 0; i < 2; ++i) {
    CUDA_CHECK(cudaMallocHost(&staging[i], kChunk));
    CUDA_CHECK(cudaEventCreateWithFlags(&done[i], cudaEventDisableTiming));
  }
  std::uint64_t pos = 0;
  for (int k = 0; pos < bytes; ++k) {
    const int b = k & 1;
    CUDA_CHECK(cudaEventSynchronize(done[b]));
    const std::size_t n = std::size_t(std::min<std::uint64_t>(kChunk, bytes - pos));
    std::size_t got = 0;
    while (got < n) {
      const ssize_t r = ::pread(fd_, static_cast<char*>(staging[b]) + got, n - got, off_t(offset + pos + got));
      if (r <= 0) throw std::runtime_error("gguf: read failed in " + path_.string());
      got += std::size_t(r);
    }
    CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(dst) + pos, staging[b], n, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(done[b], stream));
    pos += n;
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (int i = 0; i < 2; ++i) {
    cudaEventDestroy(done[i]);
    cudaFreeHost(staging[i]);
  }
}

}  // namespace ingine::model
