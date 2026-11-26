#include "model/suffix_lookup.h"

#include <array>
#include <cstdio>
#include <vector>

using ingine::model::SuffixLookup;

int main() {
  constexpr int vocab = 100000;
  SuffixLookup lookup;
  std::vector<std::int32_t> history{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3};
  const auto full = lookup.find(history, 4, 7, vocab);
  const auto limited = lookup.find(history, 4, 6, vocab);
  bool ok = full.count == 7 && full.tokens == std::array<std::int32_t, 7>{5, 6, 7, 8, 9, 10, 11} &&
            limited.count == 3 && limited.tokens == std::array<std::int32_t, 7>{5, 6, 7, 0, 0, 0, 0};
  ok &= lookup.find(history, 4, 2, vocab).count == 0;
  ok &= lookup.find(history, 40, 7, vocab).count == 0;

  // Newly accepted target tokens supply nearer matches on subsequent calls.
  history.insert(history.end(), {4, 20, 21, 22, 23, 24, 25, 26, 1, 2, 3});
  const auto recent = lookup.find(history, 4, 7, vocab);
  ok &= recent.count == 7 && recent.tokens == std::array<std::int32_t, 7>{20, 21, 22, 23, 24, 25, 26};
  // Keep the search local to the request and its bounded recent history.
  history.resize(35000, 99);
  history.insert(history.end(), {1, 2, 3});
  ok &= lookup.find(history, 4, 7, vocab).count == 0;
  std::vector<std::int32_t> image_history{1, 2, 3, 4, 5, 6, 7, -999, 9, 10, 11, 1, 2, 3};
  SuffixLookup image_lookup;
  const auto image = image_lookup.find(image_history, 4, 7, vocab);
  ok &= image.count == 3 && image.tokens == limited.tokens;
  SuffixLookup invalid_lookup;
  image_history[4] = vocab;
  ok &= invalid_lookup.find(image_history, 4, 7, vocab).count == 0;
  std::printf("%s\n", ok ? "suffix lookup passed" : "suffix lookup failed");
  return ok ? 0 : 1;
}
