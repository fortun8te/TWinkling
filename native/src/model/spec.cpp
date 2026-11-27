#include "model/spec.h"

#include "core/device.h"
#include "kernels/draft/draft.h"
#include "model/model_kernels.h"
#include "model/suffix_lookup.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

namespace ingine::model {

namespace kd = kernels::draft;
namespace kh = kernels::head;

namespace {

std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

static_assert(SuffixLookup::Proposal{}.tokens.size() == kd::GAMMA);
static_assert(sizeof(VerifyRecord::drafts) / sizeof(std::int32_t) == kd::GAMMA);

struct SuffixInput {
  std::array<std::int32_t, kd::GAMMA> tokens;
  std::array<float, kd::GAMMA> confidence;
};

Widths verify_widths(const Runtime& rt) {
  const std::span<const int> w = rt.verify_widths();
  if (w.empty() || w.size() > 8) throw std::invalid_argument("SpecLoop: the runtime has no verify widths (spec off)");
  Widths r;
  r.n = int(w.size());
  std::copy(w.begin(), w.end(), r.w);
  if (r.w[r.n - 1] < kd::BLOCK) throw std::invalid_argument("SpecLoop: the widest verify call must hold a block");
  return r;
}

}  // namespace

SpecLoop::SpecLoop(Runtime& rt, Drafter& drafter, DFlash2* flash)
    : rt_(rt), drafter_(drafter), flash_(flash), width_(sizeof(std::int32_t)), coupled_seed_(sizeof(std::uint64_t)),
      record_(sizeof(VerifyRecord)), forced_window_(kd::BLOCK * sizeof(std::int32_t)) {
  if (drafter.stream() != rt.stream()) throw std::invalid_argument("SpecLoop: drafter and runtime streams differ");
  const Widths widths = verify_widths(rt);
  // the draft outputs live at fixed addresses: one draft names them (its scratch rows are reset)
  draft_ = drafter.draft(0, 0, 0.f);
  drafter.reset();
  width_.fill(0);
  auto capture_verify = [&](const float* confidence, const std::int32_t* proposals, float threshold) {
    return rt.capture_switch(
      [&](cudaStream_t s, cudaGraphConditionalHandle h) {
        verify_select(confidence, proposals, threshold, widths, rt_.tokens(), static_cast<std::int32_t*>(width_.p),
                      h, s);
      },
      [&](cudaStream_t s, int m) {
        rt_.record_save_verify(m, s);
        rt_.record_step(m, s);
      });
  };
  verify_ = capture_verify(draft_.conf, draft_.tokens, rt.options().spec_threshold);
  if (flash_) flash_verify_ = capture_verify(flash_->confidence(), flash_->tokens(), rt.options().spec_threshold);
  if (rt.options().suffix_draft) {
    suffix_input_ = MappedHostBuffer(sizeof(SuffixInput));
    const auto* input = static_cast<const SuffixInput*>(suffix_input_.device_ptr());
    suffix_verify_ = capture_verify(input->confidence.data(), input->tokens.data(), 0.5f);
  }
  // where a round's outcome lives on the device: after the acceptance the drafter commits the accepted
  // rows and drafts for the next anchor from these (one graph, no host round trip in between)
  {
    const Drafter::AcceptScratch& scratch = drafter.accept_scratch();
    kd::RoundInputs round;
    round.nwidths = widths.n;
    for (int i = 0; i < widths.n; ++i) {
      round.widths[i] = widths.w[i];
      round.taps[i] = ptr<const __nv_bfloat16>(rt.buffers(widths.w[i]).taps);
    }
    round.m = static_cast<const std::int32_t*>(width_.p);
    round.accepted = &scratch.result->accepted;
    round.next_greedy = &scratch.result->next;
    round.next_sampled = &static_cast<VerifyRecord*>(record_.device_ptr())->next;  // sample_from_probs
    drafter.bind_round(round);
  }
  CUDA_CHECK(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming));
  CUDA_CHECK(cudaStreamSynchronize(rt.stream()));
}

SpecLoop::~SpecLoop() {
  if (done_) cudaEventDestroy(done_);
}

void SpecLoop::prepare_accept(const Sampling& sampling) {
  const bool forced = !sampling.forced.empty();
  const bool greedy = sampling.temperature <= 0.f || forced;
  const AcceptKey key = greedy ? AcceptKey{true, forced, 0.f, 1.f, 0}
                               : AcceptKey{false, false, sampling.temperature, sampling.top_p, sampling.top_k};
  if (accept_ && key == accept_key_) return;
  const Widths widths = verify_widths(rt_);
  const int V = drafter_.vocab();
  const Drafter::AcceptScratch& scratch = drafter_.accept_scratch();
  const kh::SamplingParams params{sampling.temperature, sampling.top_k, sampling.top_p};
  const auto* width = static_cast<const std::int32_t*>(width_.p);
  auto* record = static_cast<VerifyRecord*>(record_.device_ptr());
  // forced replay: the drafts are compared with the forced tokens (the argmax still runs, for the timing)
  const auto* target = forced ? static_cast<const std::int32_t*>(forced_window_.device_ptr()) : scratch.target;
  accept_ = GraphExec();
  suffix_accept_ = GraphExec();
  auto capture_accept = [&](const std::int32_t* proposals) {
    return rt_.capture_switch(
      [&](cudaStream_t s, cudaGraphConditionalHandle h) { width_select(width, widths, h, s); },
      [&](cudaStream_t s, int m) {
        // engine._accept over the m verify rows (anchor + m - 1 drafts)
        const float* logits = static_cast<const float*>(rt_.buffers(m).logits.data);
        const int n = m - 1;
        if (greedy) {
          kh::argmax(logits, m, V, V, scratch.target, s);
          kd::greedy_accept(target, proposals, n, scratch.result, s);
        } else {
          kh::sampling_probs(logits, m, V, V, params, scratch.probs, s);
          kd::sampled_accept(scratch.probs, draft_.logits, V, draft_.tokens, n, sampling.temperature, draft_.qstats,
                             scratch.uniforms, scratch.ws, scratch.resid, scratch.result, s);
        }
        rt_.record_rollback(m, &scratch.result->accepted, s);
        verify_publish(width, &scratch.result->accepted, greedy ? &scratch.result->next : nullptr, rt_.tokens(), record,
                       s);
      });
  };
  accept_ = capture_accept(draft_.tokens);
  flash_accept_ = GraphExec();
  if (flash_ && greedy) flash_accept_ = capture_accept(flash_->tokens());
  if (greedy && !forced && rt_.options().suffix_draft) {
    const auto* input = static_cast<const SuffixInput*>(suffix_input_.device_ptr());
    suffix_accept_ = capture_accept(input->tokens.data());
  }
  accept_key_ = key;
}

void SpecLoop::generate(std::int32_t token, int max_new_tokens, const Sampling& sampling, kh::SamplerRng& rng,
                        std::span<const std::int32_t> stop_ids,
                        const std::function<bool(std::span<const std::int32_t>)>& on_tokens, GenerateStats& stats,
                        std::vector<std::int32_t>& tokens_in_state) {
  if (flash_ && (sampling.temperature <= 0.f || !sampling.forced.empty())) {
    generate_flash(token, max_new_tokens, sampling, stop_ids, on_tokens, stats, tokens_in_state);
    return;
  }
  const cudaStream_t stream = rt_.stream();
  int P = int(tokens_in_state.size());  // position of the anchor (sampled, not yet processed)
  if (drafter_.length() != P) throw std::logic_error("SpecLoop: drafter context does not match the state");

  const bool forced = !sampling.forced.empty();
  const bool greedy = sampling.temperature <= 0.f || forced;  // forced: acceptance by equality
  const int V = drafter_.vocab();
  const Drafter::AcceptScratch& scratch = drafter_.accept_scratch();
  auto* record_dev = static_cast<VerifyRecord*>(record_.device_ptr());
  const auto* record = static_cast<const VerifyRecord*>(record_.data());
  prepare_accept(sampling);
  if (sampling.temperature > 0.f) {
    const std::uint64_t seed = splitmix64(rng.seed ^ splitmix64(rng.counter++));
    drafter_.set_seed(seed);
    CUDA_CHECK(cudaMemcpyAsync(coupled_seed_.p, &seed, sizeof(seed), cudaMemcpyHostToDevice, stream));
  }

  const bool lookup_enabled = rt_.options().suffix_draft && greedy && !forced;
  SuffixLookup lookup;
  constexpr int kSuffixPause = 8;
  int suffix_rounds = 0, suffix_accepted = 0, try_suffix_after = 0, misses = 0;
  bool draft_ready = !lookup_enabled;
  int emitted = 0, rounds = 0, accepted = 0;
  std::vector<std::int32_t> out{token};
  // the last verify call (none before the first round): its position, width and accepted drafts
  int round_P = -1, round_m = 0, round_acc = 0;
  // Per-round events are recorded only during forced replay or explicit diagnostic runs.
  const bool trace_rounds = forced || rt_.options().profile_spec;
  struct RoundEvents {
    cudaEvent_t e[3];
  };
  std::vector<RoundEvents> events;
  auto record_event = [&](int i) {
    CUDA_CHECK(cudaEventCreate(&events.back().e[i]));
    CUDA_CHECK(cudaEventRecord(events.back().e[i], stream));
  };
  // the first draft; later ones are enqueued by the previous round (they leave the target state alone, so
  // the last one is wasted when the generation ends) and run while the host emits
  if (draft_ready) drafter_.draft(token, P, sampling.temperature);
  for (;;) {
    const std::int32_t anchor = out.back();

    // emit this round's tokens up to a stop token or the budget
    std::size_t k = 0;
    bool done = false;
    while (k < out.size()) {
      const std::int32_t tok = out[k++];
      ++emitted;
      if (std::find(stop_ids.begin(), stop_ids.end(), tok) != stop_ids.end() || emitted == max_new_tokens) {
        done = true;
        break;
      }
    }
    stats.generated_tokens = emitted;
    const bool go_on = on_tokens(std::span<const std::int32_t>(out.data(), k));
    drafter_.note_tokens(std::span<const std::int32_t>(out.data(), k));  // for the drafts after the next one
    if (done || !go_on) {
      // The last emitted token stays unprocessed, as in plain decoding. If the generation ended inside the
      // round's accepted drafts (out[k - 1] with k - 1 < accepted), the state, the drafter and
      // tokens_in_state hold drafts past it: keep the verify call's first k rows (anchor + k - 1 drafts).
      const int keep = int(k) - 1;
      if (round_P >= 0 && keep < round_acc) {
        rt_.rollback(round_P, keep + 1, round_m);
        drafter_.commit(ptr<const __nv_bfloat16>(rt_.buffers(round_m).taps), keep + 1, round_P);
        tokens_in_state.resize(tokens_in_state.size() - std::size_t(round_acc - keep));
      }
      break;
    }

    // verify the confident drafts in the width the device picks, accept, roll back the rejected rows; then
    // commit the accepted rows and draft for the next anchor, all from device values
    bool suffix_hit = false;
    if (lookup_enabled) {
      if (rounds >= try_suffix_after) {
        const int limit = std::min(max_new_tokens - emitted, rt_.options().max_ctx - P - 1);
        const auto proposal = lookup.find(tokens_in_state, anchor, limit, V);
        if (proposal.count != 0) {
          auto* mapped = static_cast<SuffixInput*>(suffix_input_.data());
          mapped->tokens = proposal.tokens;
          for (int i = 0; i < kd::GAMMA; ++i) mapped->confidence[i] = i < proposal.count ? 1.f : 0.f;
          suffix_hit = true;
          misses = 0;
        } else if (++misses == 4) {
          try_suffix_after = rounds + 1 + kSuffixPause;
          misses = 0;
        }
      }
      if (!suffix_hit && !draft_ready) drafter_.draft(anchor, P, sampling.temperature);
    }
    const std::int32_t inputs[1] = {anchor};
    rt_.set_inputs(inputs, P);
    if (forced) {  // the forced tokens after the anchor (forced[emitted - 1]); the accept graph reads them
      auto* w = static_cast<std::int32_t*>(forced_window_.data());
      for (int i = 0; i < kd::BLOCK; ++i)
        w[i] = std::size_t(emitted + i) < sampling.forced.size() ? sampling.forced[emitted + i] : 0;
    }
    if (trace_rounds) {
      events.emplace_back();
      record_event(0);
    }
    if (!greedy) kh::uniform(kd::GAMMA, rng, scratch.uniforms, stream);
    (suffix_hit ? suffix_verify_ : verify_).launch(stream);
    (suffix_hit ? suffix_accept_ : accept_).launch(stream);
    if (!greedy) kh::sample_from_probs(scratch.resid, 1, V, V, rng, &record_dev->next, stream);
    const bool trace_experts = rt_.options().profile_experts;
    if (trace_experts) rt_.stage_expert_routes();
    CUDA_CHECK(cudaEventRecord(done_, stream));
    if (trace_rounds) record_event(1);
    // A cold lookup path goes back to the normal pipelined DSpark path
    // during a pause. Only the next active lookup needs a host decision.
    draft_ready = !lookup_enabled || rounds + 1 < try_suffix_after;
    if (draft_ready) drafter_.advance(P);
    if (trace_rounds) record_event(2);
    CUDA_CHECK(cudaEventSynchronize(done_));  // the round's only host synchronisation
    VerifyRecord r;
    std::memcpy(&r, record, sizeof r);
    const int acc = r.accepted;
    const int commit = acc + 1;
    if (trace_experts) rt_.count_expert_routes(r.m, commit, stats.expert_routes);
    round_P = P;
    round_m = r.m;
    round_acc = acc;
    if (trace_rounds) {
      int all = 0;
      while (forced && all < kd::GAMMA && std::size_t(emitted + all) < sampling.forced.size() &&
             r.drafts[all] == sampling.forced[emitted + all])
        ++all;
      stats.spec_trace.push_back({r.m, acc, all, 0.f, 0.f});
    }

    if (draft_ready) {
      drafter_.advanced(commit);
    } else {
      // Keep the draft model warm for lookup misses without predicting
      // until we know whether the next suffix has a match.
      drafter_.commit(ptr<const __nv_bfloat16>(rt_.buffers(r.m).taps), commit, P);
    }
    if (suffix_hit) {
      ++suffix_rounds;
      suffix_accepted += acc;
      if (acc < 2) try_suffix_after = rounds + 1 + kSuffixPause;
    }
    tokens_in_state.push_back(anchor);
    tokens_in_state.insert(tokens_in_state.end(), r.drafts, r.drafts + acc);
    P += commit;
    ++rounds;
    accepted += acc;
    out.assign(r.drafts, r.drafts + acc);
    out.push_back(r.next);
  }
  if (trace_rounds) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (std::size_t i = 0; i < events.size(); ++i) {
      if (i < stats.spec_trace.size()) {
        CUDA_CHECK(cudaEventElapsedTime(&stats.spec_trace[i].verify_ms, events[i].e[0], events[i].e[1]));
        CUDA_CHECK(cudaEventElapsedTime(&stats.spec_trace[i].draft_ms, events[i].e[1], events[i].e[2]));
      }
      for (cudaEvent_t e : events[i].e) cudaEventDestroy(e);
    }
  }
  stats.spec_rounds = rounds;
  stats.spec_accepted = accepted;
  stats.spec_suffix_rounds = suffix_rounds;
  stats.spec_suffix_accepted = suffix_accepted;
}

// DFlash2 rounds (greedy or forced replay): commit the previous round's kept rows, propose 7 drafts for the
// anchor (one captured DFlash2 graph), verify the confident ones in the width the device picks, accept by
// equality and roll back the rest. One host synchronisation per round, after the acceptance.
void SpecLoop::generate_flash(std::int32_t token, int max_new_tokens, const Sampling& sampling,
                              std::span<const std::int32_t> stop_ids,
                              const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
                              GenerateStats& stats, std::vector<std::int32_t>& tokens_in_state) {
  const cudaStream_t stream = rt_.stream();
  int P = int(tokens_in_state.size());
  if (flash_->length() != P) throw std::logic_error("SpecLoop: DFlash2 context does not match the state");
  const bool forced = !sampling.forced.empty();
  prepare_accept(sampling);
  const auto* record = static_cast<const VerifyRecord*>(record_.data());
  const bool trace_rounds = forced || rt_.options().profile_spec;
  struct RoundEvents {
    cudaEvent_t e[3];
  };
  std::vector<RoundEvents> events;
  auto record_event = [&](int i) {
    CUDA_CHECK(cudaEventCreate(&events.back().e[i]));
    CUDA_CHECK(cudaEventRecord(events.back().e[i], stream));
  };
  int emitted = 0, rounds = 0, accepted = 0;
  std::vector<std::int32_t> out{token};
  int round_P = -1, round_m = 0, round_acc = 0;
  for (;;) {
    const std::int32_t anchor = out.back();
    std::size_t k = 0;
    bool done = false;
    while (k < out.size()) {
      const std::int32_t tok = out[k++];
      ++emitted;
      if (std::find(stop_ids.begin(), stop_ids.end(), tok) != stop_ids.end() || emitted == max_new_tokens) {
        done = true;
        break;
      }
    }
    stats.generated_tokens = emitted;
    const bool go_on = on_tokens(std::span<const std::int32_t>(out.data(), k));
    if (done || !go_on) {
      const int keep = int(k) - 1;
      int kept = round_acc + 1;
      if (round_P >= 0 && keep < round_acc) {
        rt_.rollback(round_P, keep + 1, round_m);
        tokens_in_state.resize(tokens_in_state.size() - std::size_t(round_acc - keep));
        kept = keep + 1;
      }
      if (round_P >= 0) flash_->commit(ptr<const __nv_bfloat16>(rt_.buffers(round_m).taps), kept, round_P, true);
      break;
    }
    if (trace_rounds) {
      events.emplace_back();
      record_event(0);
    }
    if (round_P >= 0)
      flash_->commit(ptr<const __nv_bfloat16>(rt_.buffers(round_m).taps), round_acc + 1, round_P, true);
    flash_->propose(anchor);
    if (trace_rounds) record_event(1);
    const std::int32_t inputs[1] = {anchor};
    rt_.set_inputs(inputs, P);
    if (forced) {
      auto* w = static_cast<std::int32_t*>(forced_window_.data());
      for (int i = 0; i < kd::BLOCK; ++i)
        w[i] = std::size_t(emitted + i) < sampling.forced.size() ? sampling.forced[emitted + i] : 0;
    }
    flash_verify_.launch(stream);
    flash_accept_.launch(stream);
    CUDA_CHECK(cudaEventRecord(done_, stream));
    if (trace_rounds) record_event(2);
    CUDA_CHECK(cudaEventSynchronize(done_));
    VerifyRecord r;
    std::memcpy(&r, record, sizeof r);
    const int acc = r.accepted;
    round_P = P;
    round_m = r.m;
    round_acc = acc;
    if (trace_rounds) {
      int all = 0;
      while (forced && all < kd::GAMMA && std::size_t(emitted + all) < sampling.forced.size() &&
             r.drafts[all] == sampling.forced[emitted + all])
        ++all;
      stats.spec_trace.push_back({r.m, acc, all, 0.f, 0.f});
    }
    tokens_in_state.push_back(anchor);
    tokens_in_state.insert(tokens_in_state.end(), r.drafts, r.drafts + acc);
    P += acc + 1;
    ++rounds;
    accepted += acc;
    out.assign(r.drafts, r.drafts + acc);
    out.push_back(r.next);
  }
  if (trace_rounds) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (std::size_t i = 0; i < events.size(); ++i) {
      if (i < stats.spec_trace.size()) {
        CUDA_CHECK(cudaEventElapsedTime(&stats.spec_trace[i].draft_ms, events[i].e[0], events[i].e[1]));
        CUDA_CHECK(cudaEventElapsedTime(&stats.spec_trace[i].verify_ms, events[i].e[1], events[i].e[2]));
      }
      for (cudaEvent_t e : events[i].e) cudaEventDestroy(e);
    }
  }
  stats.spec_rounds = rounds;
  stats.spec_accepted = accepted;
}

}  // namespace ingine::model
