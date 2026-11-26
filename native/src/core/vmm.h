// RTX 5080 port: one contiguous device virtual range whose leading part is backed by VRAM and whose tail by
// pinned host memory (CUDA virtual memory management, CU_MEM_LOCATION_TYPE_HOST). Kernels index it like any
// device array; reads of the host-backed tail cross PCIe (about 53 GB/s on a PCIe 5.0 x16 RTX 5080 under WSL2).
// Used for routed experts: hot experts first (VRAM), then cold ones (host), pruned ones not mapped at all.
//
// The driver functions are resolved at run time (cudaGetDriverEntryPointByVersion), so nothing links
// libcuda directly (the private toolkit has no stub library).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ingine {

class TieredBuffer {
 public:
  TieredBuffer() noexcept = default;
  // `bytes` of virtual range (rounded up to the 2 MiB granularity); the first `device_bytes` (rounded up,
  // at most the range) in VRAM, the rest in host memory. Throws on failure. Contents are undefined.
  TieredBuffer(std::size_t bytes, std::size_t device_bytes);
  ~TieredBuffer();
  TieredBuffer(const TieredBuffer&) = delete;
  TieredBuffer& operator=(const TieredBuffer&) = delete;
  TieredBuffer(TieredBuffer&& o) noexcept;
  TieredBuffer& operator=(TieredBuffer&& o) noexcept;

  void* data() const noexcept { return reinterpret_cast<void*>(va_); }
  std::size_t size() const noexcept { return size_; }
  std::size_t device_bytes() const noexcept { return dev_; }
  std::size_t host_bytes() const noexcept { return size_ - dev_; }

  // Bytes of the allocation granularity (2 MiB).
  static std::size_t granularity();

 private:
  void release() noexcept;
  std::uint64_t va_ = 0;
  std::size_t size_ = 0, dev_ = 0;
  std::uint64_t handles_[2] = {0, 0};
};

// A device-only virtual range with a stable address. grow() maps pages up to the requested
// byte offset without remapping existing pages. The caller orders use and rollback against
// its compute stream. Newly mapped contents are undefined.
class GrowableBuffer {
 public:
  GrowableBuffer() noexcept = default;
  explicit GrowableBuffer(std::size_t capacity);
  ~GrowableBuffer();
  GrowableBuffer(const GrowableBuffer&) = delete;
  GrowableBuffer& operator=(const GrowableBuffer&) = delete;
  GrowableBuffer(GrowableBuffer&& other) noexcept;
  GrowableBuffer& operator=(GrowableBuffer&& other) noexcept;

  void* data() const noexcept { return reinterpret_cast<void*>(va_); }
  std::size_t mapped_bytes() const noexcept { return mapped_; }
  void grow(std::size_t bytes);
  // Undo a growth to a previous mapped_bytes() value. No kernel may still use removed pages.
  void rollback(std::size_t previous) noexcept;

 private:
  struct Chunk {
    std::size_t bytes;
    std::uint64_t handle;
  };
  void release() noexcept;
  std::uint64_t va_ = 0;
  std::size_t capacity_ = 0, reserved_ = 0, mapped_ = 0, granularity_ = 0;
  int device_ = 0;
  std::vector<Chunk> chunks_;
};

}  // namespace ingine
