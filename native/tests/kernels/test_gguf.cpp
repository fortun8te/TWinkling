// GGUF expert mode checks for one layer of a GSQ-RCO pack against the NVFP4 checkpoint:
//   1. layout: GGUF gate row c / up row I + c match NVFP4 W13 rows 2c / 2c + 1; down rows match W2 rows
//      (cosine of dequantized rows; the swapped pairing is printed as a control)
//   2. moe_up_gguf for one token against FP64 dots of the dequantized rows
//   3. gguf_to_nvfp4 against the dequantized GGUF rows
// usage: kernels_gguf GGUF MODEL_DIR LAYER
#include "kernels/moe/moe.h"
#include "core/safetensors.h"
#include "model/gguf.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace ingine;
namespace km = ingine::kernels::moe;

#define CK(x)                                                                         \
  do {                                                                                \
    cudaError_t e_ = (x);                                                             \
    if (e_ != cudaSuccess) {                                                          \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      std::exit(2);                                                                   \
    }                                                                                 \
  } while (0)

constexpr int H = 2304, I = 1152, E = 128;
static const float kE2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

double cosine(const float* a, const float* b, int n) {
  double ab = 0, aa = 0, bb = 0;
  for (int i = 0; i < n; ++i) ab += double(a[i]) * b[i], aa += double(a[i]) * a[i], bb += double(b[i]) * b[i];
  return ab / std::sqrt(aa * bb + 1e-300);
}

// NVFP4 checkpoint rows of expert e: codes [E, R, K/2], scales e4m3 [E, R, K/16], factor
std::vector<float> nv_rows(const SafetensorsModel& m, const std::string& base, int e, int R, int K, float factor) {
  const SafetensorsRef cr = m.at(base), sr = m.at(base + "_scale");
  const SafetensorsFile& f = *cr.file;
  const auto& c = *cr;
  const auto& s = *sr;
  std::vector<std::uint8_t> codes(std::size_t(R) * K / 2), sc(std::size_t(R) * K / 16);
  f.read_range(c.offset + std::uint64_t(e) * codes.size(), codes.size(), codes.data());
  f.read_range(s.offset + std::uint64_t(e) * sc.size(), sc.size(), sc.data());
  std::vector<float> out(std::size_t(R) * K);
  for (std::size_t b = 0; b < sc.size(); ++b) {
    __nv_fp8_e4m3 v;
    std::memcpy(&v, &sc[b], 1);
    const float scale = float(v);
    for (int i = 0; i < 8; ++i) {
      const std::uint8_t byte = codes[b * 8 + i];
      out[b * 16 + 2 * i] = kE2M1[byte & 0xF] * scale * factor;
      out[b * 16 + 2 * i + 1] = kE2M1[byte >> 4] * scale * factor;
    }
  }
  return out;
}

float host_scalar(const SafetensorsModel& m, const std::string& name, int i) {
  const SafetensorsRef r = m.at(name);
  const SafetensorsFile& f = *r.file;
  const auto& t = *r;
  const std::size_t n = t.bytes / 4;
  std::vector<float> v(n);
  f.read_range(t.offset, t.bytes, v.data());
  return v[n == 1 ? 0 : i];
}

std::vector<float> gguf_rows(const model::GgufFile& g, const model::GgufTensorInfo& t, std::size_t row_bytes, int K,
                             int e, int R, void* dev_w, float* dev_out, int* dev_rows) {
  g.read_to_device(t.offset + std::uint64_t(e) * R * row_bytes, std::uint64_t(R) * row_bytes, dev_w, nullptr);
  std::vector<int> rows(R);
  for (int i = 0; i < R; ++i) rows[i] = i;
  CK(cudaMemcpy(dev_rows, rows.data(), R * 4, cudaMemcpyHostToDevice));
  km::gguf_dequant_rows(t.type, dev_w, row_bytes, K, dev_rows, R, dev_out, nullptr);
  std::vector<float> out(std::size_t(R) * K);
  CK(cudaMemcpy(out.data(), dev_out, out.size() * 4, cudaMemcpyDeviceToHost));
  return out;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s GGUF MODEL_DIR LAYER\n", argv[0]);
    return 64;
  }
  const model::GgufFile g(argv[1]);
  const int L = std::atoi(argv[3]);
  const SafetensorsModel nvf(argv[2]);
  const std::string b = "blk." + std::to_string(L) + ".";
  const auto& gu = g.at(b + "ffn_gate_up_exps.weight");
  const auto& dn = g.at(b + "ffn_down_exps.weight");
  const std::size_t r13 = km::gguf_row_bytes(gu.type, H), r2 = km::gguf_row_bytes(dn.type, I);
  std::printf("layer %d: gate_up type %d (%zu B/row), down type %d (%zu B/row)\n", L, gu.type, r13, dn.type, r2);
  void *dw, *dout;
  int* drows;
  CK(cudaMalloc(&dw, std::size_t(2 * I) * r13 + H * r2));
  CK(cudaMalloc(&dout, std::size_t(2 * I) * H * 4));
  CK(cudaMalloc(&drows, 2 * I * 4));
  bool ok = true;
  const std::string nb = "model.llm.layers." + std::to_string(L) + ".mlp.experts.";
  for (int e : {0, 17, 101}) {
    const std::vector<float> g13 = gguf_rows(g, gu, r13, H, e, 2 * I, dw, static_cast<float*>(dout), drows);
    const std::vector<float> g2 = gguf_rows(g, dn, r2, I, e, H, dw, static_cast<float*>(dout), drows);
    const float f13 = host_scalar(nvf, nb + "w13_weight_amax", e) / host_scalar(nvf, nb + "w13_weight_outer_scale_denominator", e);
    const float f2 = host_scalar(nvf, nb + "w2_weight_amax", e) / host_scalar(nvf, nb + "w2_weight_outer_scale_denominator", e);
    const std::vector<float> n13 = nv_rows(nvf, nb + "w13_weight", e, 2 * I, H, f13);
    const std::vector<float> n2 = nv_rows(nvf, nb + "w2_weight", e, H, I, f2);
    double gate = 0, up = 0, swapped = 0, down = 0;
    for (int c = 0; c < I; ++c) {
      gate += cosine(&g13[std::size_t(c) * H], &n13[std::size_t(2 * c) * H], H);
      up += cosine(&g13[std::size_t(I + c) * H], &n13[std::size_t(2 * c + 1) * H], H);
      swapped += cosine(&g13[std::size_t(c) * H], &n13[std::size_t(2 * c + 1) * H], H);
    }
    for (int r = 0; r < H; ++r) down += cosine(&g2[std::size_t(r) * I], &n2[std::size_t(r) * I], I);
    gate /= I, up /= I, swapped /= I, down /= H;
    std::printf("expert %3d: mean row cosine gate %.4f up %.4f (swapped %.4f) down %.4f\n", e, gate, up, swapped, down);
    ok &= gate > 0.8 && up > 0.8 && down > 0.8 && std::fabs(swapped) < 0.1;  // low-bit rows: layout check

    // moe_up_gguf for one token routed to experts {e, ...}: slot 0 vs FP64 dots
    std::mt19937 rng(e);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<__nv_bfloat16> x(H);
    std::vector<float> xf(H);
    for (int i = 0; i < H; ++i) x[i] = __float2bfloat16(nd(rng)), xf[i] = __bfloat162float(x[i]);
    void *dx, *dq, *dg, *dact;
    int* didx;
    CK(cudaMalloc(&dx, H * 2));
    CK(cudaMalloc(&dq, km::q8_1_bytes(1, H)));
    CK(cudaMalloc(&dg, std::size_t(E) * 2 * I * r13));
    CK(cudaMalloc(&dact, 8 * I * 2));
    CK(cudaMalloc(&didx, 8 * 4));
    g.read_to_device(gu.offset, std::uint64_t(E) * 2 * I * r13, dg, nullptr);
    const int idx[8] = {e, (e + 1) % E, (e + 2) % E, (e + 3) % E, (e + 4) % E, (e + 5) % E, (e + 6) % E, (e + 7) % E};
    CK(cudaMemcpy(didx, idx, 32, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, x.data(), H * 2, cudaMemcpyHostToDevice));
    km::quantize_q8_1(static_cast<const __nv_bfloat16*>(dx), 1, H, dq, nullptr);
    km::moe_up_gguf(dq, 1, didx, gu.type, dg, r13, static_cast<__nv_bfloat16*>(dact), nullptr);
    std::vector<__nv_bfloat16> act(8 * I);
    CK(cudaMemcpy(act.data(), dact, act.size() * 2, cudaMemcpyDeviceToHost));
    double err = 0, mag = 0;
    for (int c = 0; c < I; ++c) {
      double ga = 0, ua = 0;
      for (int k = 0; k < H; ++k) ga += double(g13[std::size_t(c) * H + k]) * xf[k], ua += double(g13[std::size_t(I + c) * H + k]) * xf[k];
      const double a = std::tanh(ga / 4) * 4 * (1 / (1 + std::exp(-ga))) * (std::tanh(ua / 25) * 25);
      const double got = __bfloat162float(act[c]);
      err += (got - a) * (got - a), mag += a * a;
    }
    const double rel = std::sqrt(err / (mag + 1e-30));
    std::printf("            moe_up_gguf slot 0 relative L2 error %.4f (Q8_1 activations)\n", rel);
    ok &= rel < 0.03;

    {  // moe_down_gguf stage 1 rows for one token whose 8 slots all route to expert e
      std::vector<__nv_bfloat16> a(8 * I);
      std::vector<float> af(I);
      for (int i = 0; i < I; ++i) a[i] = __float2bfloat16(nd(rng)), af[i] = __bfloat162float(a[i]);
      for (int i = I; i < 8 * I; ++i) a[i] = a[i % I];
      void *da, *daq, *dd2, *dys, *dw8, *dcw, *dst, *dout;
      CK(cudaMalloc(&da, 8 * I * 2));
      CK(cudaMalloc(&daq, km::q8_1_bytes(8, I)));
      CK(cudaMalloc(&dd2, std::size_t(E) * H * r2));
      CK(cudaMalloc(&dys, km::down_scratch_bytes()));
      CK(cudaMalloc(&dw8, 32));
      CK(cudaMalloc(&dcw, H * 16));
      CK(cudaMalloc(&dst, 3 * H * 4));
      CK(cudaMalloc(&dout, H * 4));
      CK(cudaMemset(dw8, 0, 32));
      CK(cudaMemset(dcw, 0, H * 16));
      CK(cudaMemset(dst, 0, 3 * H * 4));
      g.read_to_device(dn.offset, std::uint64_t(E) * H * r2, dd2, nullptr);
      const int same[8] = {e, e, e, e, e, e, e, e};
      CK(cudaMemcpy(didx, same, 32, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(da, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
      km::quantize_q8_1(static_cast<const __nv_bfloat16*>(da), 8, I, daq, nullptr);
      km::moe_down_gguf(daq, 1, didx, static_cast<float*>(dw8), dn.type, dd2, r2, static_cast<float*>(dys),
                        static_cast<float*>(dcw), static_cast<float*>(dst), static_cast<float*>(dout), nullptr, nullptr);
      std::vector<float> yv(H);
      CK(cudaMemcpy(yv.data(), dys, H * 4, cudaMemcpyDeviceToHost));  // y[slot 0][token 0][:]
      double de = 0, dm = 0;
      for (int r = 0; r < H; ++r) {
        double ref = 0;
        for (int k = 0; k < I; ++k) ref += double(g2[std::size_t(r) * I + k]) * af[k];
        de += (yv[r] - ref) * (yv[r] - ref), dm += ref * ref;
      }
      const double drel = std::sqrt(de / (dm + 1e-30));
      std::printf("            moe_down_gguf rows relative L2 error %.4f\n", drel);
      ok &= drel < 0.03;
      for (void* p : {da, daq, dd2, dys, dw8, dcw, dst, dout}) cudaFree(p);
    }
    // transcode to NVFP4 and compare with the dequantized GGUF rows of expert e
    void *c13, *s13, *f13d, *am;
    CK(cudaMalloc(&c13, std::size_t(E) * 2 * I * H / 2));
    CK(cudaMalloc(&s13, std::size_t(E) * 2 * I * H / 16));
    CK(cudaMalloc(&f13d, E * 4));
    CK(cudaMalloc(&am, E * 4));
    km::gguf_to_nvfp4(gu.type, dg, r13, 2 * I, H, E, true, am, c13, s13, static_cast<float*>(f13d), nullptr);
    std::vector<std::uint8_t> hc(std::size_t(2 * I) * H / 2), hs(std::size_t(2 * I) * H / 16);
    float hf;
    CK(cudaMemcpy(hc.data(), static_cast<char*>(c13) + std::size_t(e) * hc.size(), hc.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hs.data(), static_cast<char*>(s13) + std::size_t(e) * hs.size(), hs.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&hf, static_cast<float*>(f13d) + e, 4, cudaMemcpyDeviceToHost));
    double tc = 0;
    std::vector<float> row(H);
    for (int r = 0; r < 2 * I; ++r) {
      for (int k = 0; k < H; ++k) {
        const std::size_t blk = std::size_t(r) * (H / 16) + k / 16;
        __nv_fp8_e4m3 v;
        std::memcpy(&v, &hs[blk], 1);
        const std::uint8_t byte = hc[std::size_t(r) * (H / 2) + k / 2];
        row[k] = kE2M1[(k & 1) ? byte >> 4 : byte & 0xF] * float(v) * hf;
      }
      const int src = (r & 1) ? I + r / 2 : r / 2;
      tc += cosine(row.data(), &g13[std::size_t(src) * H], H);
    }
    tc /= 2 * I;
    std::printf("            gguf_to_nvfp4 mean row cosine %.4f\n", tc);
    ok &= tc > 0.97;
    for (void* p : {dx, dq, dg, dact, static_cast<void*>(didx), c13, s13, f13d, am}) cudaFree(p);
  }

  if (std::getenv("GGUF_BENCH")) {
    // decode cost per call at widths 1, 2, 4, 8: tokens routed to 8 random experts each
    void *dg, *d2, *dq, *daq, *dact, *dwts, *dconv, *dstate, *dout;
    int* didx;
    CK(cudaMalloc(&dg, std::size_t(E) * 2 * I * r13));
    CK(cudaMalloc(&d2, std::size_t(E) * H * r2));
    g.read_to_device(gu.offset, std::uint64_t(E) * 2 * I * r13, dg, nullptr);
    g.read_to_device(dn.offset, std::uint64_t(E) * H * r2, d2, nullptr);
    CK(cudaMalloc(&dq, km::q8_1_bytes(8, H)));
    CK(cudaMalloc(&daq, km::q8_1_bytes(64, I)));
    CK(cudaMalloc(&dact, 64 * I * 2));
    CK(cudaMalloc(&dwts, 64 * 4));
    CK(cudaMalloc(&dconv, H * 16));
    CK(cudaMalloc(&dstate, 3 * H * 4));
    CK(cudaMalloc(&dout, 8 * H * 4));
    CK(cudaMalloc(&didx, 64 * 4));
    void* dys;
    CK(cudaMalloc(&dys, km::down_scratch_bytes()));
    CK(cudaMemset(dq, 0, km::q8_1_bytes(8, H)));
    CK(cudaMemset(daq, 0, km::q8_1_bytes(64, I)));
    CK(cudaMemset(dwts, 0, 256));
    CK(cudaMemset(dconv, 0, H * 16));
    CK(cudaMemset(dstate, 0, 3 * H * 4));
    std::mt19937 rng(7);
    cudaEvent_t a, z;
    cudaEventCreate(&a), cudaEventCreate(&z);
    for (int M : {1, 2, 4, 8}) {
      std::vector<int> idx(64);
      for (int n = 0; n < M; ++n) {
        std::vector<int> pick(E);
        for (int i = 0; i < E; ++i) pick[i] = i;
        std::shuffle(pick.begin(), pick.end(), rng);
        for (int k = 0; k < 8; ++k) idx[n * 8 + k] = pick[k];
      }
      std::vector<int> u(idx.begin(), idx.begin() + 8 * M);
      std::sort(u.begin(), u.end());
      const int uniq = int(std::unique(u.begin(), u.end()) - u.begin());
      CK(cudaMemcpy(didx, idx.data(), 256, cudaMemcpyHostToDevice));
      for (int rep = 0; rep < 3; ++rep) {
        km::moe_up_gguf(dq, M, didx, gu.type, dg, r13, static_cast<__nv_bfloat16*>(dact), nullptr);
        km::moe_down_gguf(daq, M, didx, static_cast<float*>(dwts), dn.type, d2, r2, static_cast<float*>(dys), static_cast<float*>(dconv),
                          static_cast<float*>(dstate), static_cast<float*>(dout), nullptr, nullptr);
      }
      constexpr int N = 50;
      cudaEventRecord(a);
      for (int i = 0; i < N; ++i) km::moe_up_gguf(dq, M, didx, gu.type, dg, r13, static_cast<__nv_bfloat16*>(dact), nullptr);
      cudaEventRecord(z);
      cudaEventSynchronize(z);
      float up_ms;
      cudaEventElapsedTime(&up_ms, a, z);
      cudaEventRecord(a);
      for (int i = 0; i < N; ++i)
        km::moe_down_gguf(daq, M, didx, static_cast<float*>(dwts), dn.type, d2, r2, static_cast<float*>(dys), static_cast<float*>(dconv),
                          static_cast<float*>(dstate), static_cast<float*>(dout), nullptr, nullptr);
      cudaEventRecord(z);
      cudaEventSynchronize(z);
      float dn_ms;
      cudaEventElapsedTime(&dn_ms, a, z);
      const double up_us = up_ms * 1000 / N, dn_us = dn_ms * 1000 / N;
      const double up_gb = double(uniq) * 2 * I * r13 / (up_us * 1e3), dn_gb = double(uniq) * H * r2 / (dn_us * 1e3);
      std::printf("bench M=%d unique=%d: up %.1f us (%.0f GB/s)  down %.1f us (%.0f GB/s)\n", M, uniq, up_us, up_gb, dn_us, dn_gb);
    }
  }
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
