#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>

namespace ingine::model {

// Session-local prompt lookup for greedy speculative decoding. Every proposed
// token is checked by the target. Keep recent exact four-token suffixes that
// have enough known continuation to justify the target's width-four graph.
class SuffixLookup {
 public:
  struct Proposal {
    std::array<std::int32_t, 7> tokens{};
    int count = 0;
  };

  Proposal find(std::span<const std::int32_t> history, std::int32_t anchor, int limit, int vocab) {
    if (history.size() < kKeySize || limit < kMinDraft) return {};
    const std::size_t first = history.size() - std::min(history.size(), kHistory);
    // History grows by accepted target tokens. Occasionally discard the old
    // index in one pass so long generations cannot grow it without bound.
    if (next_ > history.size() || first >= base_ + kHistory) {
      latest_.clear();
      next_ = 0;
    }
    if (next_ == 0) base_ = first;
    next_ = std::max(next_, first + kKeySize);
    for (; next_ + kMinDraft <= history.size(); ++next_) {
      const auto i = next_ - kKeySize;
      latest_[{history[i], history[i + 1], history[i + 2], history[i + 3]}] = next_;
    }
    const auto n = history.size();
    const auto match = latest_.find({history[n - 3], history[n - 2], history[n - 1], anchor});
    if (match == latest_.end() || match->second - kKeySize < first) return {};
    const auto end = match->second;
    int count = limit >= 7 && n - end >= 7 ? 7 : kMinDraft;
    for (int i = 0; i < count; ++i) {
      // Multimodal history can contain image-only ids outside the model's
      // output vocabulary. Avoid inserting those into target verify rows.
      const auto token = history[end + i];
      if (token < 0 || token >= vocab) {
        if (i < kMinDraft) return {};
        count = kMinDraft;
      }
    }
    Proposal proposal;
    proposal.count = count;
    std::copy_n(history.begin() + end, count, proposal.tokens.begin());
    return proposal;
  }

 private:
  static constexpr std::size_t kKeySize = 4, kHistory = 32768;
  static constexpr int kMinDraft = 3;
  using Key = std::array<std::int32_t, kKeySize>;
  struct Hash {
    std::size_t operator()(const Key& key) const noexcept {
      std::uint64_t h = 0xcbf29ce484222325ull;
      for (const auto token : key) h = (h ^ std::uint32_t(token)) * 0x100000001b3ull;
      return std::size_t(h);
    }
  };
  std::unordered_map<Key, std::size_t, Hash> latest_;
  std::size_t next_ = 0, base_ = 0;
};

}  // namespace ingine::model
