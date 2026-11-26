// RTX 5080 port: see vmm.h.
#include "core/vmm.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace ingine {
namespace {

struct Driver {
  CUresult (*mem_create)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long) = nullptr;
  CUresult (*mem_release)(CUmemGenericAllocationHandle) = nullptr;
  CUresult (*address_reserve)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long) = nullptr;
  CUresult (*address_free)(CUdeviceptr, size_t) = nullptr;
  CUresult (*map)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long) = nullptr;
  CUresult (*unmap)(CUdeviceptr, size_t) = nullptr;
  CUresult (*set_access)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) = nullptr;
  CUresult (*granularity)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags) = nullptr;
};

template <class F>
void resolve(const char* name, F& fn) {
  void* p = nullptr;
  cudaDriverEntryPointQueryResult q{};
  if (cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q) != cudaSuccess ||
      q != cudaDriverEntryPointSuccess || p == nullptr)
    throw std::runtime_error(std::string("TieredBuffer: cannot resolve ") + name);
  fn = reinterpret_cast<F>(p);
}

const Driver& driver() {
  static Driver d;
  static std::once_flag once;
  std::call_once(once, [] {
    cudaFree(nullptr);  // the primary context
    resolve("cuMemCreate", d.mem_create);
    resolve("cuMemRelease", d.mem_release);
    resolve("cuMemAddressReserve", d.address_reserve);
    resolve("cuMemAddressFree", d.address_free);
    resolve("cuMemMap", d.map);
    resolve("cuMemUnmap", d.unmap);
    resolve("cuMemSetAccess", d.set_access);
    resolve("cuMemGetAllocationGranularity", d.granularity);
  });
  return d;
}

int current_device() {
  int dev = 0;
  cudaGetDevice(&dev);
  return dev;
}

CUmemAllocationProp prop(bool host) {
  CUmemAllocationProp p{};
  p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  p.location.type = host ? CU_MEM_LOCATION_TYPE_HOST : CU_MEM_LOCATION_TYPE_DEVICE;
  p.location.id = host ? 0 : current_device();
  return p;
}

void check(CUresult r, const char* what) {
  if (r != CUDA_SUCCESS) throw std::runtime_error(std::string("TieredBuffer: ") + what + " failed (" + std::to_string(int(r)) + ")");
}

}  // namespace

std::size_t TieredBuffer::granularity() {
  const Driver& d = driver();
  const CUmemAllocationProp p = prop(false);
  size_t g = 0;
  check(d.granularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM), "cuMemGetAllocationGranularity");
  const CUmemAllocationProp h = prop(true);
  size_t gh = 0;
  check(d.granularity(&gh, &h, CU_MEM_ALLOC_GRANULARITY_MINIMUM), "cuMemGetAllocationGranularity (host)");
  return std::max(g, gh);
}

TieredBuffer::TieredBuffer(std::size_t bytes, std::size_t device_bytes) {
  if (bytes == 0) return;
  const Driver& d = driver();
  const std::size_t g = granularity();
  size_ = (bytes + g - 1) / g * g;
  dev_ = std::min(size_, (device_bytes + g - 1) / g * g);
  CUdeviceptr va = 0;
  check(d.address_reserve(&va, size_, g, 0, 0), "cuMemAddressReserve");
  va_ = va;
  try {
    if (dev_ > 0) {
      const CUmemAllocationProp p = prop(false);
      CUmemGenericAllocationHandle h = 0;
      check(d.mem_create(&h, dev_, &p, 0), "cuMemCreate (device)");
      handles_[0] = h;
      check(d.map(va, dev_, 0, h, 0), "cuMemMap (device)");
    }
    if (size_ > dev_) {
      const CUmemAllocationProp p = prop(true);
      CUmemGenericAllocationHandle h = 0;
      check(d.mem_create(&h, size_ - dev_, &p, 0), "cuMemCreate (host)");
      handles_[1] = h;
      check(d.map(va + dev_, size_ - dev_, 0, h, 0), "cuMemMap (host)");
    }
    CUmemAccessDesc a{};
    a.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    a.location.id = current_device();
    a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    check(d.set_access(va, size_, &a, 1), "cuMemSetAccess");
  } catch (...) {
    release();
    throw;
  }
}

void TieredBuffer::release() noexcept {
  if (va_ == 0) return;
  const Driver& d = driver();
  cudaDeviceSynchronize();
  if (handles_[0]) d.unmap(va_, dev_);
  if (handles_[1]) d.unmap(va_ + dev_, size_ - dev_);
  for (std::uint64_t& h : handles_)
    if (h) d.mem_release(h), h = 0;
  d.address_free(va_, size_);
  va_ = 0;
  size_ = dev_ = 0;
}

TieredBuffer::~TieredBuffer() { release(); }

TieredBuffer::TieredBuffer(TieredBuffer&& o) noexcept
    : va_(std::exchange(o.va_, 0)), size_(std::exchange(o.size_, 0)), dev_(std::exchange(o.dev_, 0)) {
  handles_[0] = std::exchange(o.handles_[0], 0);
  handles_[1] = std::exchange(o.handles_[1], 0);
}

TieredBuffer& TieredBuffer::operator=(TieredBuffer&& o) noexcept {
  if (this != &o) {
    release();
    va_ = std::exchange(o.va_, 0);
    size_ = std::exchange(o.size_, 0);
    dev_ = std::exchange(o.dev_, 0);
    handles_[0] = std::exchange(o.handles_[0], 0);
    handles_[1] = std::exchange(o.handles_[1], 0);
  }
  return *this;
}

GrowableBuffer::GrowableBuffer(std::size_t capacity) : capacity_(capacity), device_(current_device()) {
  if (capacity == 0) return;
  const Driver& d = driver();
  const CUmemAllocationProp p = prop(false);
  check(d.granularity(&granularity_, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
        "cuMemGetAllocationGranularity (elastic KV)");
  if (!granularity_ || capacity > std::numeric_limits<std::size_t>::max() - (granularity_ - 1))
    throw std::invalid_argument("elastic KV capacity cannot be page aligned");
  reserved_ = (capacity + (granularity_ - 1)) / granularity_ * granularity_;
  CUdeviceptr va = 0;
  check(d.address_reserve(&va, reserved_, granularity_, 0, 0), "cuMemAddressReserve (elastic KV)");
  va_ = va;
}

void GrowableBuffer::grow(std::size_t bytes) {
  if (bytes > capacity_) throw std::invalid_argument("elastic KV growth exceeds its reserved capacity");
  if (bytes <= mapped_) return;
  if (current_device() != device_) throw std::runtime_error("elastic KV must grow on its original CUDA device");
  const std::size_t end = (bytes + (granularity_ - 1)) / granularity_ * granularity_;
  const std::size_t size = end - mapped_;
  const CUdeviceptr base = va_ + mapped_;
  const Driver& d = driver();
  const CUmemAllocationProp p = prop(false);
  // Allocate bookkeeping first so no throwing host allocation follows a successful CUDA mapping.
  chunks_.reserve(chunks_.size() + 1);
  CUmemGenericAllocationHandle handle = 0;
  check(d.mem_create(&handle, size, &p, 0), "cuMemCreate (elastic KV)");
  bool mapped = false;
  try {
    check(d.map(base, size, 0, handle, 0), "cuMemMap (elastic KV)");
    mapped = true;
    CUmemAccessDesc a{};
    a.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    a.location.id = device_;
    a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    check(d.set_access(base, size, &a, 1), "cuMemSetAccess (elastic KV)");
  } catch (...) {
    if (mapped) d.unmap(base, size);
    d.mem_release(handle);
    throw;
  }
  chunks_.push_back({size, handle});
  mapped_ = end;
}

void GrowableBuffer::rollback(std::size_t previous) noexcept {
  if (previous >= mapped_) return;
  const Driver& d = driver();
  while (!chunks_.empty()) {
    const Chunk chunk = chunks_.back();
    const std::size_t start = mapped_ - chunk.bytes;
    if (start < previous || d.unmap(va_ + start, chunk.bytes) != CUDA_SUCCESS) return;
    d.mem_release(chunk.handle);
    chunks_.pop_back();
    mapped_ = start;
  }
}

void GrowableBuffer::release() noexcept {
  if (!va_) return;
  const Driver& d = driver();
  rollback(0);
  d.address_free(va_, reserved_);
  va_ = 0;
  capacity_ = reserved_ = mapped_ = granularity_ = 0;
}

GrowableBuffer::~GrowableBuffer() { release(); }

GrowableBuffer::GrowableBuffer(GrowableBuffer&& other) noexcept { *this = std::move(other); }

GrowableBuffer& GrowableBuffer::operator=(GrowableBuffer&& other) noexcept {
  if (this != &other) {
    release();
    va_ = std::exchange(other.va_, 0);
    capacity_ = std::exchange(other.capacity_, 0);
    reserved_ = std::exchange(other.reserved_, 0);
    granularity_ = std::exchange(other.granularity_, 0);
    mapped_ = std::exchange(other.mapped_, 0);
    device_ = other.device_;
    chunks_ = std::move(other.chunks_);
  }
  return *this;
}

}  // namespace ingine
