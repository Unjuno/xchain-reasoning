// SPDX-License-Identifier: Apache-2.0
// Scalar, single-thread, synchronous BP. No fast-math and no query labels.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>
#include <new>

extern "C" int xchain_bp(int n, int e, int b, int q,
    const int* src, const int* dst, const int* rev,
    const double* jc, const double* fields, int depth, int mode,
    unsigned long long random_seed, double* output, double* belief1,
    double* belief2, int* selected) {
  if (n < 1 || n > 256 || e < 1 || e > 65536 || b < 1 ||
      q < 0 || q >= n || depth < 2 || depth > 256 || mode < 0 || mode > 3)
    return 1;
  for (int a = 0; a < e; ++a) {
    if (src[a] < 0 || src[a] >= n || dst[a] < 0 || dst[a] >= n ||
        rev[a] < 0 || rev[a] >= e || !std::isfinite(jc[a])) return 2;
  }
  try {
    std::vector<double> tj(e), msg(e), next(e), total(n);
    std::vector<int> qin;
    for (int a = 0; a < e; ++a) {
      tj[a] = std::tanh(jc[a]);
      if (dst[a] == q) qin.push_back(a);
    }
    auto sweep = [&](const double* h) {
      std::fill(total.begin(), total.end(), 0.0);
      for (int a = 0; a < e; ++a) total[dst[a]] += msg[a];
      for (int a = 0; a < e; ++a) {
        const double cavity = (h[src[a]] + total[src[a]]) - msg[rev[a]];
        const double z = std::max(-1.0 + 1e-14,
                            std::min(1.0 - 1e-14, tj[a] * std::tanh(cavity)));
        next[a] = std::atanh(z);
      }
      msg.swap(next);
    };
    auto read = [&](const double* h) {
      double answer = h[q];
      for (int a : qin) answer += msg[a];
      return answer;
    };
    std::vector<double> saved;
    if (mode >= 2) saved.resize(static_cast<std::size_t>(b) * e);
    int k = 0;
    for (int s = 0; s < b; ++s) {
      const double* h = fields + static_cast<std::size_t>(s) * n;
      std::fill(msg.begin(), msg.end(), 0.0);
      sweep(h); belief1[s] = read(h);
      sweep(h); belief2[s] = read(h);
      selected[s] = (belief1[s] >= 0.0) != (belief2[s] >= 0.0);
      k += selected[s];
      if (mode >= 2) {
        std::copy(msg.begin(), msg.end(), saved.begin() + static_cast<std::size_t>(s) * e);
        output[s] = belief2[s];
      } else {
        if (mode == 0 || selected[s]) {
          for (int t = 2; t < depth; ++t) sweep(h);
        }
        output[s] = read(h);
      }
    }
    if (mode >= 2) {
      std::vector<int> order(b);
      std::iota(order.begin(), order.end(), 0);
      if (mode == 2) {
        std::stable_sort(order.begin(), order.end(), [&](int a, int c) {
          return std::abs(belief2[a]) < std::abs(belief2[c]);
        });
      } else {
        std::mt19937_64 rng(random_seed);
        std::shuffle(order.begin(), order.end(), rng);
      }
      order.resize(k);
      std::sort(order.begin(), order.end());
      std::fill(selected, selected + b, 0);
      for (int s : order) {
        selected[s] = 1;
        const double* h = fields + static_cast<std::size_t>(s) * n;
        std::copy(saved.begin() + static_cast<std::size_t>(s) * e,
                  saved.begin() + static_cast<std::size_t>(s + 1) * e, msg.begin());
        for (int t = 2; t < depth; ++t) sweep(h);
        output[s] = read(h);
      }
    }
  } catch (const std::bad_alloc&) { return 3; }
  catch (...) { return 4; }
  return 0;
}
