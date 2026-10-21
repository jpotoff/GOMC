/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
// Exactness test for CBMC with one-step look-ahead (CBMC_LookAhead), as used
// by DCLinkedHedron.
//
// At step i, k trials are drawn with full weights w_j. For each trial a
// look-ahead estimate v_j = mean of m fresh samples of the next step's weight
// given trial j is computed. A trial is selected with probability
// w_j v_j / sum_l w_l v_l and the step contributes
//     W_i = (sum_j w_j v_j) / (k v_c).
// The look-ahead sets are auxiliary variables drawn with their trials, so the
// scheme is exact although v is random. The old configuration fills slot 0
// with its actual trial; every slot, slot 0 included, gets a fresh
// look-ahead set.
//
// The toy chain is the one in RecoilGrowthTest: n steps of D states,
// non-uniform proposal, look-ahead energy structure and hard closures. The
// stationary distribution of whole-chain moves must match full enumeration.
// Broken variants of the weight rule run as negative controls and must fail.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace {

const double INF = std::numeric_limits<double>::infinity();

struct Model {
  uint n = 5, D = 4;
  double q[4] = {0.4, 0.3, 0.2, 0.1};
  double b[4] = {1.0, 0.6, 1.5, 0.4};
  double e1[4][4] = {{0.5, -1.0, INF, 1.5},
                     {2.0, 0.0, -0.5, INF},
                     {-1.2, INF, 1.0, 0.3},
                     {INF, 0.8, -0.3, 2.5}};
  double e2[4][4] = {{INF, 0.4, 0.0, -0.6},
                     {0.2, INF, 1.1, 0.0},
                     {0.0, -0.8, 3.0, INF},
                     {1.3, 0.0, INF, 0.5}};
  double A(int s) const { return b[s] / q[s]; }
  double U(const std::vector<int> &path, uint step, int s) const {
    double u = 0.0;
    if (step >= 1)
      u += e1[path[step - 1]][s];
    if (step >= 2)
      u += e2[path[step - 2]][s];
    return u;
  }
  double W(const std::vector<int> &path, uint step, int s) const {
    double u = U(path, step, s);
    return u == INF ? 0.0 : A(s) * std::exp(-u);
  }
};

enum Variant {
  CORRECT,
  NO_DIVIDE_V,   // W_i = sum_j w_j v_j / k (forgets 1/v_c)
  PLAIN_WEIGHT,  // selects with w v but weighs as plain CBMC, sum_j w_j / k
  OLD_REUSES_V   // old slot 0 uses the actual next state as its look-ahead
};

class LookAheadChain {
public:
  LookAheadChain(const Model &m, uint k, uint la, Variant v, uint seed)
      : model(m), k(k), m(la), variant(v), gen(seed), uni(0.0, 1.0) {}

  double Grow(std::vector<int> &path) {
    path.clear();
    double W = 1.0;
    std::vector<int> s(k);
    std::vector<double> w(k), v(k), sel(k);
    for (uint i = 0; i < model.n; ++i) {
      double S = 0.0, sumW = 0.0;
      for (uint j = 0; j < k; ++j) {
        s[j] = Draw();
        w[j] = model.W(path, i, s[j]);
        v[j] = LookAhead(path, i, s[j]);
        sel[j] = w[j] * v[j];
        S += sel[j];
        sumW += w[j];
      }
      if (!(S > 0.0))
        return 0.0;
      double x = uni(gen) * S, c = 0.0;
      uint pick = k - 1;
      for (uint j = 0; j < k; ++j) {
        c += sel[j];
        if (x < c) {
          pick = j;
          break;
        }
      }
      W *= Factor(S, sumW, v[pick]);
      path.push_back(s[pick]);
    }
    return W;
  }

  double Weigh(const std::vector<int> &actual) {
    std::vector<int> path;
    double W = 1.0;
    for (uint i = 0; i < model.n; ++i) {
      double S = 0.0, sumW = 0.0, v0 = 0.0;
      for (uint j = 0; j < k; ++j) {
        int s = j == 0 ? actual[i] : Draw();
        double w = model.W(path, i, s);
        double v = LookAhead(path, i, s);
        if (j == 0 && variant == OLD_REUSES_V)
          v = i + 1 < model.n ? model.W(Extend(path, s), i + 1, actual[i + 1])
                              : 1.0;
        if (j == 0)
          v0 = v;
        S += w * v;
        sumW += w;
      }
      if (!(v0 > 0.0))
        return INF; // the reverse move could not select the actual trial
      W *= Factor(S, sumW, v0);
      path.push_back(actual[i]);
    }
    return W;
  }

  double Uniform() { return uni(gen); }

private:
  int Draw() {
    double x = uni(gen), c = 0.0;
    int s = 0;
    for (; s < (int)model.D - 1; ++s) {
      c += model.q[s];
      if (x < c)
        break;
    }
    return s;
  }
  std::vector<int> Extend(const std::vector<int> &path, int s) const {
    std::vector<int> p = path;
    p.push_back(s);
    return p;
  }
  // mean weight of m fresh samples of the next step given trial s at step i
  double LookAhead(const std::vector<int> &path, uint i, int s) {
    if (i + 1 >= model.n || m == 0) // m = 0: plain CBMC
      return 1.0;
    std::vector<int> p = Extend(path, s);
    double sum = 0.0;
    for (uint l = 0; l < m; ++l)
      sum += model.W(p, i + 1, Draw());
    return sum / m;
  }
  double Factor(double S, double sumW, double vc) const {
    switch (variant) {
    case NO_DIVIDE_V:
      return S / k;
    case PLAIN_WEIGHT:
      return sumW / k;
    default:
      return S / (k * vc);
    }
  }

  const Model &model;
  uint k, m;
  Variant variant;
  std::mt19937_64 gen;
  std::uniform_real_distribution<double> uni;
};

std::vector<double> Exact(const Model &m) {
  uint N = 1;
  for (uint i = 0; i < m.n; ++i)
    N *= m.D;
  std::vector<double> pi(N, 0.0);
  std::vector<int> path(m.n);
  double z = 0.0;
  for (uint idx = 0; idx < N; ++idx) {
    uint x = idx;
    for (uint i = m.n; i-- > 0;) {
      path[i] = x % m.D;
      x /= m.D;
    }
    double w = 1.0;
    for (uint i = 0; i < m.n && w > 0.0; ++i)
      w *= m.W(path, i, path[i]) * m.q[path[i]];
    pi[idx] = w;
    z += w;
  }
  for (double &p : pi)
    p /= z;
  return pi;
}

struct ChainResult {
  double acc, maxZ;
};

// Markov chain of whole-chain moves; largest batch-means |z| over states
// with pi >= 0.002 against the exact distribution
ChainResult RunChain(const Model &m, uint k, uint la, Variant v, uint moves,
                     uint seed) {
  LookAheadChain chain(m, k, la, v, seed);
  std::vector<double> pi = Exact(m);
  uint start = std::max_element(pi.begin(), pi.end()) - pi.begin();
  std::vector<int> cur(m.n);
  for (uint i = m.n, x = start; i-- > 0; x /= m.D)
    cur[i] = x % m.D;
  const uint B = 20, perBatch = moves / B;
  std::vector<std::vector<double>> hist(B, std::vector<double>(pi.size(), 0.0));
  uint accepted = 0;
  std::vector<int> trial;
  for (uint it = 0; it < B * perBatch; ++it) {
    double wn = chain.Grow(trial);
    double wo = chain.Weigh(cur);
    if (wn > 0.0 && wo < INF && chain.Uniform() * wo < wn) {
      cur = trial;
      ++accepted;
    }
    uint idx = 0;
    for (int s : cur)
      idx = idx * m.D + s;
    hist[it / perBatch][idx] += 1.0;
  }
  ChainResult r;
  r.acc = (double)accepted / (B * perBatch);
  r.maxZ = 0.0;
  for (size_t i = 0; i < pi.size(); ++i) {
    double mean = 0.0, sq = 0.0;
    for (uint b = 0; b < B; ++b) {
      double f = hist[b][i] / perBatch;
      mean += f;
      sq += f * f;
    }
    mean /= B;
    double var = (sq / B - mean * mean) * B / (B - 1);
    if (pi[i] >= 0.002 && var > 0.0)
      r.maxZ = std::max(r.maxZ, std::fabs(mean - pi[i]) / std::sqrt(var / B));
  }
  return r;
}

} // namespace

TEST(LookAheadCBMCTest, MatchesExactDistribution) {
  Model m;
  // m = 0 is plain CBMC (v = 1)
  const uint cases[][2] = {{4, 0}, {1, 1}, {2, 1}, {4, 2}, {4, 8}, {8, 4}};
  for (const auto &c : cases) {
    ChainResult r = RunChain(m, c[0], c[1], CORRECT, 2000000, 31 + c[0] * 7 + c[1]);
    // correct rules give max|z| ~ 2-3 (one 1M-move run reached 6.1); the
    // broken variants below give 18-57
    EXPECT_LT(r.maxZ, 6.0) << "k=" << c[0] << " m=" << c[1];
    std::cout << "[ LA      ] k=" << c[0] << " m=" << c[1] << "  acceptance "
              << r.acc << "  max|z| " << r.maxZ << "\n";
  }
}

TEST(LookAheadCBMCTest, BrokenWeightsAreDetected) {
  // negative controls: the test must be able to see a wrong weight rule
  Model m;
  const Variant broken[] = {NO_DIVIDE_V, PLAIN_WEIGHT, OLD_REUSES_V};
  const char *names[] = {"no 1/v_c", "plain CBMC weight", "old reuses v"};
  for (int b = 0; b < 3; ++b) {
    ChainResult r = RunChain(m, 4, 2, broken[b], 1000000, 101 + b);
    EXPECT_GT(r.maxZ, 8.0) << names[b];
    std::cout << "[ LA      ] broken: " << names[b] << "  max|z| " << r.maxZ
              << "\n";
  }
}

// Note: look-ahead is not guaranteed to raise acceptance. W_i divides by v_c,
// so a noisy look-ahead estimate adds variance: on this toy, k = 4 accepts
// 18.7% with plain CBMC, 17.7% with m = 8 and 5.9% with m = 2.
