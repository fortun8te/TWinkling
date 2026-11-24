// RTX 5080 port: see vmm.h.
#include "core/vmm.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
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

}  // namespace ingine
