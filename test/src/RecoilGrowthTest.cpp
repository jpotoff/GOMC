/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
// Exactness test for the recoil growth engine (src/cbmc/RecoilGrowth.h).
//
// A toy chain of n steps, each choosing one of D states. Trials are drawn
// from a non-uniform proposal q, so each carries a bonded importance weight
// a(s) = b(s) / q(s), as GOMC's trials do. The step energy depends on the
// previous two states and includes hard closures (infinite energy), so some
// partial chains are dead ends a step or two later -- the situation recoil
// growth is meant for. The target is
//     pi(path) ~ prod_i b(s_i) exp(-u_i(path)).
// A Markov chain of whole-chain recoil-growth moves, accepted with
// min(1, W_new / W_old), must reproduce pi exactly; it is compared with full
// enumeration for several trial counts k and recoil lengths l.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "RecoilGrowth.h"

namespace {

const double INF = std::numeric_limits<double>::infinity();

struct Model {
  uint n = 5, D = 4;
  double q[4] = {0.4, 0.3, 0.2, 0.1};
  double b[4] = {1.0, 0.6, 1.5, 0.4};
  // energy of state s after previous state p (index [p][s])
  double e1[4][4] = {{0.5, -1.0, INF, 1.5},
                     {2.0, 0.0, -0.5, INF},
                     {-1.2, INF, 1.0, 0.3},
                     {INF, 0.8, -0.3, 2.5}};
  // energy of state s two steps after state pp (look-ahead structure)
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
};

struct ToyTrial {
  int s;
  double w, p;
};

struct Uniform {
  explicit Uniform(std::mt19937_64 &g) : gen(g), dist(0.0, 1.0) {}
  double operator()() { return dist(gen); }
  std::mt19937_64 &gen;
  std::uniform_real_distribution<double> dist;
};

class ToyChain {
public:
  typedef ToyTrial Trial;
  // trials with energy below theta (in kT) are always open
  ToyChain(const Model &m, Uniform &u, double theta)
      : model(m), uni(u), theta(theta) {}

  uint NumSteps() const { return model.n; }
  void Generate(uint step, uint n, Trial *out) {
    EXPECT_EQ(path.size(), step) << "trials generated off the placed path";
    for (uint t = 0; t < n; ++t) {
      double x = uni(), c = 0.0;
      int s = 0;
      for (; s < (int)model.D - 1; ++s) {
        c += model.q[s];
        if (x < c)
          break;
      }
      Fill(step, s, out[t]);
    }
  }
  void Actual(uint step, Trial &out) { Fill(step, actual[step], out); }
  void Place(uint step, const Trial &t) {
    EXPECT_EQ(path.size(), step) << "placement out of order";
    path.push_back(t.s);
  }
  void Unplace(uint step) {
    EXPECT_EQ(path.size(), step + 1) << "unplace out of order";
    path.pop_back();
  }

  std::vector<int> path, actual;

private:
  void Fill(uint step, int s, Trial &t) {
    double u = model.U(path, step, s);
    t.s = s;
    if (u == INF) {
      t.w = t.p = 0.0;
      return;
    }
    t.w = model.A(s) * std::exp(-u);
    t.p = std::min(1.0, std::exp(theta - u));
  }
  const Model &model;
  Uniform &uni;
  double theta;
};

uint Index(const std::vector<int> &path, uint D) {
  uint idx = 0;
  for (int s : path)
    idx = idx * D + s;
  return idx;
}

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
    double logw = 0.0;
    for (uint i = 0; i < m.n; ++i) {
      double u = m.U(path, i, path[i]);
      if (u == INF) {
        logw = -INF;
        break;
      }
      logw += std::log(m.b[path[i]]) - u;
    }
    pi[idx] = std::exp(logw);
    z += pi[idx];
  }
  for (double &p : pi)
    p /= z;
  return pi;
}

struct ChainResult {
  double acc;  // acceptance
  double tv;   // total variation distance from the exact distribution
  double maxZ; // largest |z| over states with pi >= 0.002, using batch-means
               // standard errors (20 batches), which account for correlation
};

ChainResult RunChain(const Model &m, uint k, uint l, double theta,
                     uint moves, uint seed) {
  std::mt19937_64 gen(seed);
  Uniform uni(gen);
  cbmc::RecoilGrowth<ToyChain, Uniform> rg(k, l);
  std::vector<double> pi = Exact(m);
  // start from the most probable chain
  uint start = std::max_element(pi.begin(), pi.end()) - pi.begin();
  std::vector<int> cur(m.n);
  for (uint i = m.n, x = start; i-- > 0; x /= m.D)
    cur[i] = x % m.D;

  const uint B = 20, perBatch = moves / B;
  std::vector<std::vector<double>> hist(B, std::vector<double>(pi.size(), 0.0));
  uint accepted = 0;
  for (uint it = 0; it < B * perBatch; ++it) {
    ToyChain nc(m, uni, theta);
    double wn = rg.GrowNew(nc, uni);
    std::vector<int> trial = nc.path;
    ToyChain oc(m, uni, theta);
    oc.actual = cur;
    double wo = rg.WeighOld(oc, uni);
    EXPECT_GT(wo, 0.0);
    EXPECT_EQ(oc.path, cur);
    if (wn > 0.0 && uni() * wo < wn) {
      EXPECT_EQ(trial.size(), m.n);
      cur = trial;
      ++accepted;
    }
    hist[it / perBatch][Index(cur, m.D)] += 1.0;
  }
  ChainResult r;
  r.acc = (double)accepted / (B * perBatch);
  r.tv = 0.0;
  r.maxZ = 0.0;
  for (size_t i = 0; i < pi.size(); ++i) {
    double mean = 0.0, sq = 0.0;
    for (uint b = 0; b < B; ++b) {
      double f = hist[b][i] / perBatch;
      mean += f;
      sq += f * f;
    }
    mean /= B;
    r.tv += 0.5 * std::fabs(mean - pi[i]);
    double var = (sq / B - mean * mean) * B / (B - 1);
    if (pi[i] >= 0.002 && var > 0.0)
      r.maxZ = std::max(r.maxZ, std::fabs(mean - pi[i]) / std::sqrt(var / B));
  }
  return r;
}

} // namespace

TEST(RecoilGrowthTest, ToyChainHasDeadEnds) {
  // the test is only meaningful if some partial chains cannot be completed
  Model m;
  std::vector<double> pi = Exact(m);
  uint nonzero = 0;
  for (double p : pi)
    nonzero += p > 0.0;
  EXPECT_GT(nonzero, 50u);
  EXPECT_LT(nonzero, pi.size() / 2);
}

TEST(RecoilGrowthTest, MatchesExactDistribution) {
  Model m;
  // k, l, open-probability threshold theta
  const double cases[][3] = {{1, 1, 0.0}, {2, 1, 0.0}, {3, 2, 0.0},
                             {4, 2, 0.0}, {2, 3, 0.0}, {4, 3, 0.0},
                             {4, 2, 1.0}, {3, 3, 1.0}};
  for (const auto &c : cases) {
    uint k = c[0], l = c[1];
    ChainResult r = RunChain(m, k, l, c[2], 1000000, 17 + k * 10 + l);
    // Deliberately broken engines fail this (see the commit that added it):
    // dropping 1/p, picking uniformly with the weighted formula, or leaving
    // the actual trial out of the old sum all give max|z| far above 5.5.
    EXPECT_LT(r.maxZ, 5.5) << "k=" << k << " l=" << l << " theta=" << c[2];
    std::cout << "[ RG      ] k=" << k << " l=" << l << " theta=" << c[2]
              << "  acceptance " << r.acc << "  TV " << r.tv << "  max|z| "
              << r.maxZ << "\n";
  }
}

TEST(RecoilGrowthTest, LookAheadRaisesAcceptance) {
  // with dead ends two steps ahead, l = 3 should accept more than l = 1
  Model m;
  double acc1 = RunChain(m, 4, 1, 0.0, 200000, 5).acc;
  double acc3 = RunChain(m, 4, 3, 0.0, 200000, 6).acc;
  EXPECT_GT(acc3, acc1);
}
