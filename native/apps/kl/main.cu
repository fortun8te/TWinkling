// ingine-kl (RTX 5080 port): quality and expert-calibration tool.
//
//   ingine-kl --data FILE.jsonl --dump REF.bin     teacher-forced log-probs of every position (width-8 decode
//                                                 graphs), the reference for --ref
//   ingine-kl --ref REF.bin [--json OUT]          the same positions on this configuration: mean/percentile
//                                                 KL(reference || this), top-1 agreement, NLL of both
//   ingine-kl --data FILE.jsonl --calib STATS.bin  routing statistics over prefill (per visit and expert:
//                                                 pairs, sum of weights, REAP saliency sum of w * ||out||)
//
// Data: JSON lines {"name", "text"} (tokenised without special tokens) or {"name", "messages", "tools"?}
// (rendered with the model's chat template, the assistant turns included). --max-tokens truncates each item.
// Model options as ingine-serve: --model-dir --ablation --expert-plan --embed-host --kv-fp8 --bf16-dense
// --max-ctx --prefill-chunk.
#include "core/arena.h"
#include "core/device.h"
#include "model/prefill.h"
#include "model/runtime.h"
#include "text/chat.h"

#include <cuda_fp16.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using ingine::DeviceBuffer;
using namespace ingine::model;

namespace {

constexpr int V = UNPADDED_VOCAB;
constexpr int kThreads = 1024;

__device__ float block_reduce(float v, bool is_max, float* sh) {
  for (int o = 16; o > 0; o >>= 1) {
    const float u = __shfl_xor_sync(0xffffffffu, v, o);
    v = is_max ? fmaxf(v, u) : v + u;
  }
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __syncthreads();
  if (lane == 0) sh[w] = v;
  __syncthreads();
  if (w == 0) {
    v = lane < (blockDim.x >> 5) ? sh[lane] : (is_max ? -INFINITY : 0.f);
    for (int o = 16; o > 0; o >>= 1) {
      const float u = __shfl_xor_sync(0xffffffffu, v, o);
      v = is_max ? fmaxf(v, u) : v + u;
    }
    if (lane == 0) sh[32] = v;
  }
  __syncthreads();
  return sh[32];
}

__device__ float row_lse(const float* x, float* sh) {
  float m = -INFINITY;
  for (int i = threadIdx.x; i < V; i += blockDim.x) m = fmaxf(m, x[i]);
  m = block_reduce(m, true, sh);
  float s = 0.f;
  for (int i = threadIdx.x; i < V; i += blockDim.x) s += expf(x[i] - m);
  s = block_reduce(s, false, sh);
  return m + logf(s);
}

// out[r] = fp16(log_softmax(logits[r]))
__global__ void logsoftmax_kernel(const float* logits, __half* out) {
  __shared__ float sh[33];
  const float* x = logits + size_t(blockIdx.x) * V;
  const float lse = row_lse(x, sh);
  for (int i = threadIdx.x; i < V; i += blockDim.x) out[size_t(blockIdx.x) * V + i] = __float2half_rn(x[i] - lse);
}

// res[r] = {KL(ref || cand), ref argmax == cand argmax, -lp_ref[target], -lp_cand[target]}
__global__ void kl_kernel(const __half* ref, const float* logits, const int* targets, float* res) {
  __shared__ float sh[33];
  __shared__ unsigned long long best[2];
  const float* x = logits + size_t(blockIdx.x) * V;
  const __half* r = ref + size_t(blockIdx.x) * V;
  const float lse = row_lse(x, sh);
  if (threadIdx.x == 0) best[0] = best[1] = 0ull;
  __syncthreads();
  float kl = 0.f;
  float bref = -INFINITY, bcand = -INFINITY;
  int iref = 0, icand = 0;
  for (int i = threadIdx.x; i < V; i += blockDim.x) {
    const float lr = __half2float(r[i]);
    const float lc = x[i] - lse;
    const float p = expf(lr);
    if (p > 0.f) kl += p * (lr - lc);
    if (lr > bref) bref = lr, iref = i;
    if (x[i] > bcand) bcand = x[i], icand = i;
  }
  kl = block_reduce(kl, false, sh);
  // argmax by (value, lowest index) via packed keys: values are finite here
  auto key = [](float v, int i) {
    unsigned u = __float_as_uint(v);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return (static_cast<unsigned long long>(u) << 32) | static_cast<unsigned>(0x7fffffff - i);
  };
  atomicMax(&best[0], key(bref, iref));
  atomicMax(&best[1], key(bcand, icand));
  __syncthreads();
  if (threadIdx.x == 0) {
    const int ar = 0x7fffffff - int(best[0] & 0xffffffffu), ac = 0x7fffffff - int(best[1] & 0xffffffffu);
    const int t = targets[blockIdx.x];
    res[4 * blockIdx.x + 0] = fmaxf(kl, 0.f);
    res[4 * blockIdx.x + 1] = ar == ac ? 1.f : 0.f;
    res[4 * blockIdx.x + 2] = -__half2float(r[t]);
    res[4 * blockIdx.x + 3] = -(x[t] - lse);
  }
}

struct Item {
  std::string name;
  std::vector<std::int32_t> ids;
};

std::vector<Item> load_data(const fs::path& path, const fs::path& model_dir, int max_tokens, int max_items) {
  const ingine::text::ChatFormat fmt = ingine::text::ChatFormat::load(model_dir);
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path.string());
  std::vector<Item> items;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (max_items > 0 && int(items.size()) >= max_items) break;
    const nlohmann::ordered_json j = nlohmann::ordered_json::parse(line);
    Item it;
    it.name = j.value("name", "item" + std::to_string(items.size()));
    try {
      if (j.contains("text")) {
        it.ids = fmt.tokenizer().encode(j.at("text").get<std::string>(), false);
      } else {
        const nlohmann::ordered_json tools = j.contains("tools") ? j.at("tools") : nlohmann::ordered_json();
        ingine::text::ChatOptions opt;
        it.ids = fmt.render(j.at("messages"), tools, opt).ids;
      }
    } catch (const std::exception& e) {
      std::fprintf(stderr, "skipping %s: %s\n", it.name.c_str(), e.what());
      continue;
    }
    if (max_tokens > 0 && int(it.ids.size()) > max_tokens) it.ids.resize(max_tokens);
    if (it.ids.size() < 16) continue;
    items.push_back(std::move(it));
  }
  return items;
}

template <class T>
void wr(std::ofstream& o, const T& v) {
  o.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <class T>
T rd(std::ifstream& i) {
  T v{};
  i.read(reinterpret_cast<char*>(&v), sizeof(T));
  if (!i) throw std::runtime_error("reference file truncated");
  return v;
}

// Teacher-forced decode of `ids` from an empty state: calls row(p, logits) for every position p < n - 1 whose
// logits (device fp32 [V]) predict ids[p + 1]; rows come in groups of up to 8 (`rows` = group size).
template <class F>
void forced(Runtime& rt, const std::vector<std::int32_t>& ids, F&& group) {
  reset_state(rt.state(), rt.stream());
  const int n = int(ids.size());
  int p = 0;
  while (p < n - 1) {
    int m = 8;
    while (m > 1 && p + m > n - 1) m /= 2;
    if (!rt.has_width(m)) m = 1;
    rt.set_inputs(std::span<const std::int32_t>(ids.data() + p, std::size_t(m)), p);
    const float* logits = rt.run_step(m);
    group(p, m, logits);
    p += m;
  }
}

double percentile(std::vector<float> v, double q) {
  if (v.empty()) return 0;
  const std::size_t k = std::min(v.size() - 1, std::size_t(q * double(v.size())));
  std::nth_element(v.begin(), v.begin() + std::ptrdiff_t(k), v.end());
  return v[k];
}

void usage() {
  std::printf(
      "usage: ingine-kl (--data FILE --dump REF | --ref REF [--json OUT] | --data FILE --calib STATS)\n"
      "                 [--model-dir DIR] [--ablation FILE] [--expert-plan FILE] [--embed-host] [--kv-fp8]\n"
      "                 [--bf16-dense] [--max-ctx N] [--max-tokens N] [--items N] [--prefill-chunk N]\n");
}

}  // namespace

int main(int argc, char** argv) {
  RuntimeOptions opt;
  opt.model_dir = std::getenv("INGINE_MODEL_DIR") ? std::getenv("INGINE_MODEL_DIR") : "";
  opt.dense_fp8 = true;
  opt.max_ctx = 0;
  opt.prefill_chunk = 2048;
  fs::path data, dump, ref, calib, json_out;
  int max_tokens = 4096, items_max = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) {
        usage();
        std::exit(2);
      }
      return argv[++i];
    };
    if (k == "--model-dir") opt.model_dir = val();
    else if (k == "--ablation") opt.ablation = val();
    else if (k == "--expert-plan") opt.expert_plan = val();
    else if (k == "--embed-host") opt.embed_host = true;
    else if (k == "--kv-fp8") opt.kv_fp8 = true;
    else if (k == "--bf16-dense") opt.dense_fp8 = false;
    else if (k == "--max-ctx") opt.max_ctx = std::stoi(val());
    else if (k == "--prefill-chunk") opt.prefill_chunk = std::stoi(val());
    else if (k == "--max-tokens") max_tokens = std::stoi(val());
    else if (k == "--items") items_max = std::stoi(val());
    else if (k == "--data") data = val();
    else if (k == "--dump") dump = val();
    else if (k == "--ref") ref = val();
    else if (k == "--calib") calib = val();
    else if (k == "--json") json_out = val();
    else {
      usage();
      return k == "--help" || k == "-h" ? 0 : 2;
    }
  }
  if (int(!dump.empty()) + int(!ref.empty()) + int(!calib.empty()) != 1 || opt.model_dir.empty()) {
    usage();
    return 2;
  }
  try {
    using Clock = std::chrono::steady_clock;
    std::vector<Item> items;
    std::ifstream rin;
    int ref_items = 0;
    if (!ref.empty()) {
      rin.open(ref, std::ios::binary);
      if (!rin) throw std::runtime_error("cannot open " + ref.string());
      char magic[4];
      rin.read(magic, 4);
      if (std::memcmp(magic, "IKL1", 4) != 0 || rd<std::int32_t>(rin) != V) throw std::runtime_error("not an ingine-kl reference");
      ref_items = rd<std::int32_t>(rin);
      max_tokens = rd<std::int32_t>(rin);
    } else {
      items = load_data(data, opt.model_dir, max_tokens, items_max);
      std::size_t total = 0;
      for (const Item& it : items) total += it.ids.size();
      std::printf("%zu items, %zu tokens\n", items.size(), total);
    }
    if (opt.max_ctx <= 0) opt.max_ctx = (std::max(max_tokens, 1024) + 1023) / 1024 * 1024;
    opt.spec = calib.empty();  // decode widths 2, 4, 8 (no drafter is loaded by a bare Runtime)
    cudaFree(nullptr);
    std::size_t free0 = 0, total0 = 0;
    cudaMemGetInfo(&free0, &total0);
    const auto t0 = Clock::now();
    Runtime rt(opt);
    std::size_t free1 = 0;
    cudaMemGetInfo(&free1, &total0);
    std::printf("loaded in %.1f s: %.0f MiB of VRAM used, %.0f MiB free\n",
                std::chrono::duration<double>(Clock::now() - t0).count(), double(free0 - free1) / 1048576.0,
                double(free1) / 1048576.0);
    std::fflush(stdout);
    const cudaStream_t s = rt.stream();

    if (!calib.empty()) {
      Prefiller pre(rt.weights(), rt.ablation(), opt.prefill_chunk, s);
      DeviceBuffer stats(std::size_t(NUM_VISITS) * EXPERTS * 3 * 4), tok(std::size_t(opt.prefill_chunk) * 4),
          logits(std::size_t(V) * 4);
      stats.fill(0);
      pre.set_stats(static_cast<float*>(stats.p));
      std::size_t done = 0, n_items = 0;
      const auto t1 = Clock::now();
      for (const Item& it : items) {
        reset_state(rt.state(), s);
        for (int p = 0; p < int(it.ids.size()); p += opt.prefill_chunk) {
          const int T = std::min(opt.prefill_chunk, int(it.ids.size()) - p);
          CUDA_CHECK(cudaMemcpyAsync(tok.p, it.ids.data() + p, std::size_t(T) * 4, cudaMemcpyHostToDevice, s));
          pre.run(rt.state(), static_cast<const std::int32_t*>(tok.p), T, p, static_cast<float*>(logits.p), nullptr);
        }
        CUDA_CHECK(cudaStreamSynchronize(s));
        done += it.ids.size();
        if (++n_items % 20 == 0) {
          const double el = std::chrono::duration<double>(Clock::now() - t1).count();
          std::printf("  %zu items, %zu tokens, %.0f tok/s\n", n_items, done, done / el);
          std::fflush(stdout);
        }
      }
      const double secs = std::chrono::duration<double>(Clock::now() - t1).count();
      std::vector<float> host(std::size_t(NUM_VISITS) * EXPERTS * 3);
      stats.copy_to_host(host.data(), host.size() * 4);
      std::ofstream o(calib, std::ios::binary);
      o.write("IKS1", 4);
      wr(o, std::int32_t(NUM_VISITS));
      wr(o, std::int32_t(EXPERTS));
      wr(o, std::int64_t(done));
      o.write(reinterpret_cast<const char*>(host.data()), std::streamsize(host.size() * 4));
      std::printf("calibration: %zu tokens in %.1f s (%.0f tok/s) -> %s\n", done, secs, done / secs, calib.c_str());
      return 0;
    }

    DeviceBuffer half_rows(std::size_t(8) * V * 2), res(std::size_t(8) * 4 * 4), targets(8 * 4);
    if (!dump.empty()) {
      std::ofstream o(dump, std::ios::binary);
      o.write("IKL1", 4);
      wr(o, std::int32_t(V));
      wr(o, std::int32_t(items.size()));
      wr(o, std::int32_t(max_tokens));
      std::vector<__half> rows;
      const auto t1 = Clock::now();
      std::size_t positions = 0;
      for (const Item& it : items) {
        const int n = int(it.ids.size());
        rows.assign(std::size_t(n - 1) * V, __half());
        forced(rt, it.ids, [&](int p, int m, const float* logits) {
          logsoftmax_kernel<<<m, kThreads, 0, s>>>(logits, static_cast<__half*>(half_rows.p));
          CUDA_CHECK(cudaMemcpyAsync(rows.data() + std::size_t(p) * V, half_rows.p, std::size_t(m) * V * 2,
                                     cudaMemcpyDeviceToHost, s));
          CUDA_CHECK(cudaStreamSynchronize(s));
        });
        wr(o, std::int32_t(it.name.size()));
        o.write(it.name.data(), std::streamsize(it.name.size()));
        wr(o, std::int32_t(n));
        o.write(reinterpret_cast<const char*>(it.ids.data()), std::streamsize(std::size_t(n) * 4));
        o.write(reinterpret_cast<const char*>(rows.data()), std::streamsize(rows.size() * 2));
        positions += std::size_t(n - 1);
        std::printf("  %-32s %6d tokens\n", it.name.c_str(), n);
        std::fflush(stdout);
      }
      const double secs = std::chrono::duration<double>(Clock::now() - t1).count();
      std::printf("reference: %zu positions in %.1f s (%.0f tok/s teacher-forced) -> %s\n", positions, secs,
                  positions / secs, dump.c_str());
      return 0;
    }

    // --ref
    nlohmann::ordered_json report = nlohmann::ordered_json::object();
    report["model_dir"] = opt.model_dir.string();
    report["expert_plan"] = opt.expert_plan.string();
    report["embed_host"] = opt.embed_host;
    report["kv_fp8"] = opt.kv_fp8;
    report["vram_used_mib"] = double(free0 - free1) / 1048576.0;
    std::vector<float> all_kl;
    double sum_kl = 0, sum_top1 = 0, sum_nr = 0, sum_nc = 0;
    std::size_t positions = 0;
    std::vector<__half> rows;
    const auto t1 = Clock::now();
    for (int k = 0; k < ref_items; ++k) {
      Item it;
      it.name.resize(std::size_t(rd<std::int32_t>(rin)));
      rin.read(it.name.data(), std::streamsize(it.name.size()));
      const int n = rd<std::int32_t>(rin);
      it.ids.resize(std::size_t(n));
      rin.read(reinterpret_cast<char*>(it.ids.data()), std::streamsize(std::size_t(n) * 4));
      rows.resize(std::size_t(n - 1) * V);
      rin.read(reinterpret_cast<char*>(rows.data()), std::streamsize(rows.size() * 2));
      if (!rin) throw std::runtime_error("reference file truncated");
      std::vector<float> host(std::size_t(n) * 4);
      forced(rt, it.ids, [&](int p, int m, const float* logits) {
        CUDA_CHECK(cudaMemcpyAsync(half_rows.p, rows.data() + std::size_t(p) * V, std::size_t(m) * V * 2,
                                   cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(targets.p, it.ids.data() + p + 1, std::size_t(m) * 4, cudaMemcpyHostToDevice, s));
        kl_kernel<<<m, kThreads, 0, s>>>(static_cast<const __half*>(half_rows.p), logits,
                                         static_cast<const int*>(targets.p), static_cast<float*>(res.p));
        CUDA_CHECK(cudaMemcpyAsync(host.data() + std::size_t(p) * 4, res.p, std::size_t(m) * 16, cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
      });
      double ikl = 0, it1 = 0, inr = 0, inc = 0;
      for (int p = 0; p < n - 1; ++p) {
        ikl += host[4 * p], it1 += host[4 * p + 1], inr += host[4 * p + 2], inc += host[4 * p + 3];
        all_kl.push_back(host[4 * p]);
      }
      const int c = n - 1;
      sum_kl += ikl, sum_top1 += it1, sum_nr += inr, sum_nc += inc;
      positions += std::size_t(c);
      std::printf("  %-32s %6d tok  KL %.4f  top1 %.2f%%  ppl ref %.3f this %.3f\n", it.name.c_str(), n, ikl / c,
                  100 * it1 / c, std::exp(inr / c), std::exp(inc / c));
      std::fflush(stdout);
      report["items"][it.name] = {{"tokens", n}, {"kl", ikl / c}, {"top1", it1 / c}, {"ppl_ref", std::exp(inr / c)},
                                  {"ppl", std::exp(inc / c)}};
    }
    const double secs = std::chrono::duration<double>(Clock::now() - t1).count();
    const double N = double(positions);
    report["positions"] = positions;
    report["kl_mean"] = sum_kl / N;
    report["kl_p90"] = percentile(all_kl, 0.90);
    report["kl_p99"] = percentile(all_kl, 0.99);
    report["kl_max"] = all_kl.empty() ? 0.0 : double(*std::max_element(all_kl.begin(), all_kl.end()));
    report["top1"] = sum_top1 / N;
    report["ppl_ref"] = std::exp(sum_nr / N);
    report["ppl"] = std::exp(sum_nc / N);
    report["seconds"] = secs;
    std::printf("TOTAL %zu positions: KL mean %.5f p90 %.4f p99 %.4f max %.3f | top1 %.2f%% | ppl ref %.4f this %.4f "
                "(%.1f s)\n",
                positions, sum_kl / N, double(report["kl_p90"]), double(report["kl_p99"]), double(report["kl_max"]),
                100 * sum_top1 / N, std::exp(sum_nr / N), std::exp(sum_nc / N), secs);
    if (!json_out.empty()) std::ofstream(json_out) << report.dump(2) << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ingine-kl: %s\n", e.what());
    return 1;
  }
}
