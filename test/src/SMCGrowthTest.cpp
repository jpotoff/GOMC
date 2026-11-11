/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
// Exactness test and toy benchmark for population (SMC) chain growth
// (src/cbmc/SMCGrowth.h).
//
// Same toy chain as RecoilGrowthTest: n steps, each choosing one of D states
// from a non-uniform proposal q, with bonded importance weight a = b / q and
// an energy that depends on the previous two states, with hard closures
// (infinite energy) that make some partial chains dead ends. The target is
//     pi(path) ~ prod_i b(s_i) exp(-u_i(path)).
// A Markov chain of whole-chain moves accepted with
// min(1, Z_hat_new / Z_hat_old), Z_hat_old from conditional SMC, must
// reproduce pi exactly.
//
// The benchmark (SMCGrowthBench, disabled by default) uses longer random
// chains, where log Z is known exactly by a transfer matrix over state pairs,
// and compares SMC with single-chain CBMC and with independent chains at an
// equal number of trials per step (N * k).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "SMCGrowth.h"

namespace {

const double INF = std::numeric_limits<double>::infinity();

struct Model {
  uint n, D;
  std::vector<double> q, b;
  std::vector<std::vector<double>> e1, e2; // [prev][s], [prev-prev][s]
  std::vector<std::vector<double>> e0;     // [step][s], position-dependent

  double A(int s) const { return b[s] / q[s]; }
  double U(const std::vector<int> &path, uint step, int s) const {
    double u = e0[step][s];
    if (step >= 1)
      u += e1[path[step - 1]][s];
    if (step >= 2)
      u += e2[path[step - 2]][s];
    return u;
  }
};

// The RecoilGrowthTest model: 5 steps, 4 states.
Model SmallModel() {
  Model m;
  m.n = 5;
  m.D = 4;
  m.q = {0.4, 0.3, 0.2, 0.1};
  m.b = {1.0, 0.6, 1.5, 0.4};
  m.e1 = {{0.5, -1.0, INF, 1.5},
          {2.0, 0.0, -0.5, INF},
          {-1.2, INF, 1.0, 0.3},
          {INF, 0.8, -0.3, 2.5}};
  m.e2 = {{INF, 0.4, 0.0, -0.6},
          {0.2, INF, 1.1, 0.0},
          {0.0, -0.8, 3.0, INF},
          {1.3, 0.0, INF, 0.5}};
  m.e0.assign(m.n, std::vector<double>(m.D, 0.0));
  return m;
}

// Random chain: pair and next-nearest energies ~ N(0, sigma) with a fraction
// `closed` of pairs forbidden, plus a position-dependent field ~ N(0, sigma)
// with a fraction `closed0` forbidden (the "environment").
Model RandomModel(uint n, uint D, double sigma, double closed, double closed0,
                  uint seed) {
  std::mt19937_64 g(seed);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  std::normal_distribution<double> G(0.0, sigma);
  Model m;
  m.n = n;
  m.D = D;
  double qs = 0.0;
  for (uint s = 0; s < D; ++s) {
    m.q.push_back(0.5 + U(g));
    qs += m.q.back();
    m.b.push_back(0.5 + U(g));
  }
  for (double &x : m.q)
    x /= qs;
  auto table = [&](uint rows, double f) {
    std::vector<std::vector<double>> t(rows, std::vector<double>(D));
    for (auto &r : t)
      for (double &x : r)
        x = U(g) < f ? INF : G(g);
    return t;
  };
  m.e1 = table(D, closed);
  m.e2 = table(D, closed);
  m.e0 = table(n, closed0);
  return m;
}

// Exact log Z by a transfer matrix over (previous, current) state pairs.
double ExactLogZ(const Model &m) {
  const uint D = m.D;
  // f[p][s] = log of summed weight of all paths ending in (p, s)
  std::vector<double> f(D * D, -INF), g(D * D);
  std::vector<double> f0(D, -INF);
  for (uint s = 0; s < D; ++s)
    if (m.e0[0][s] < INF)
      f0[s] = std::log(m.b[s]) - m.e0[0][s];
  if (m.n == 1) {
    double mx = *std::max_element(f0.begin(), f0.end()), z = 0.0;
    for (double x : f0)
      z += std::exp(x - mx);
    return mx + std::log(z);
  }
  for (uint p = 0; p < D; ++p)
    for (uint s = 0; s < D; ++s) {
      double u = m.e0[1][s] + m.e1[p][s];
      if (f0[p] > -INF && u < INF)
        f[p * D + s] = f0[p] + std::log(m.b[s]) - u;
    }
  for (uint step = 2; step < m.n; ++step) {
    for (uint p = 0; p < D; ++p)
      for (uint s = 0; s < D; ++s) {
        double mx = -INF;
        std::vector<double> terms;
        for (uint pp = 0; pp < D; ++pp) {
          double u = m.e0[step][s] + m.e1[p][s] + m.e2[pp][s];
          if (f[pp * D + p] == -INF || u == INF)
            continue;
          terms.push_back(f[pp * D + p] + std::log(m.b[s]) - u);
          mx = std::max(mx, terms.back());
        }
        if (terms.empty()) {
          g[p * D + s] = -INF;
          continue;
        }
        double z = 0.0;
        for (double t : terms)
          z += std::exp(t - mx);
        g[p * D + s] = mx + std::log(z);
      }
    f.swap(g);
  }
  double mx = *std::max_element(f.begin(), f.end()), z = 0.0;
  for (double x : f)
    if (x > -INF)
      z += std::exp(x - mx);
  return mx + std::log(z);
}

struct Uniform {
  explicit Uniform(std::mt19937_64 &g) : gen(&g), dist(0.0, 1.0) {}
  double operator()() { return dist(*gen); }
  std::mt19937_64 *gen;
  std::uniform_real_distribution<double> dist;
};

// One particle: standard CBMC with k trials per step.
class ToyParticle {
public:
  ToyParticle(const Model &m, Uniform &u, uint k,
              const std::vector<int> *act = nullptr)
      : model(&m), uni(&u), k(k), actual(act) {
    path.reserve(m.n);
  }

  uint NumSteps() const { return model->n; }
  double Extend(uint step, bool reference) {
    EXPECT_EQ(path.size(), step) << "trials generated off the placed path";
    s.resize(k);
    w.resize(k);
    uint first = 0;
    if (reference) {
      s[0] = (*actual)[step];
      w[0] = Weight(step, s[0]);
      first = 1;
    }
    double sum = 0.0;
    for (uint t = 0; t < k; ++t) {
      if (t >= first) {
        double x = (*uni)(), c = 0.0;
        int st = 0;
        for (; st < (int)model->D - 1; ++st) {
          c += model->q[st];
          if (x < c)
            break;
        }
        s[t] = st;
        w[t] = Weight(step, st);
      }
      sum += w[t];
    }
    // the reference is placed even with zero weight (see SMCGrowth)
    if (!(sum > 0.0) && !reference)
      return 0.0;
    uint pick = 0;
    if (!reference) {
      double x = (*uni)() * sum, c = 0.0;
      for (pick = 0; pick < k - 1; ++pick) {
        c += w[pick];
        if (x < c)
          break;
      }
      while (w[pick] <= 0.0 && pick > 0) // guard round-off
        --pick;
    }
    path.push_back(s[pick]);
    return sum / k;
  }

  std::vector<int> path;

private:
  double Weight(uint step, int st) const {
    double u = model->U(path, step, st);
    return (u == INF) ? 0.0 : model->A(st) * std::exp(-u);
  }
  const Model *model;
  Uniform *uni;
  uint k;
  const std::vector<int> *actual;
  std::vector<int> s;
  std::vector<double> w;
};

typedef cbmc::SMCGrowth<ToyParticle, Uniform> SMC;

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
  double acc, tv, maxZ;
};

// Markov chain of whole-chain SMC moves. `brokenOld` weighs the old
// configuration with an unconditional run (a negative control).
ChainResult RunChain(const Model &m, uint N, uint k, double tau, uint moves,
                     uint seed, bool brokenOld = false) {
  std::mt19937_64 gen(seed);
  Uniform uni(gen);
  SMC smc(N, tau);
  std::vector<double> pi = Exact(m);
  uint start = std::max_element(pi.begin(), pi.end()) - pi.begin();
  std::vector<int> cur(m.n);
  for (uint i = m.n, x = start; i-- > 0; x /= m.D)
    cur[i] = x % m.D;

  const uint B = 20, perBatch = moves / B;
  std::vector<std::vector<double>> hist(B, std::vector<double>(pi.size(), 0.0));
  uint accepted = 0;
  for (uint it = 0; it < B * perBatch; ++it) {
    double ln = smc.GrowNew(ToyParticle(m, uni, k), uni);
    std::vector<int> trial = smc.Chosen().path;
    double lo;
    if (brokenOld) {
      lo = smc.GrowNew(ToyParticle(m, uni, k), uni);
    } else {
      lo = smc.WeighOld(ToyParticle(m, uni, k, &cur), uni);
      EXPECT_GT(lo, -INF);
      EXPECT_EQ(smc.Chosen().path, cur);
    }
    if (ln > -INF && (lo == -INF || std::log(uni()) < ln - lo)) {
      EXPECT_EQ(trial.size(), m.n);
      cur = trial;
      ++accepted;
    }
    hist[it / perBatch][Index(cur, m.D)] += 1.0;
  }
  ChainResult r{(double)accepted / (B * perBatch), 0.0, 0.0};
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

TEST(SMCGrowthTest, ExactLogZMatchesEnumeration) {
  Model m = RandomModel(6, 5, 1.0, 0.25, 0.1, 3);
  uint N = 1;
  for (uint i = 0; i < m.n; ++i)
    N *= m.D;
  std::vector<int> path(m.n);
  double z = 0.0;
  for (uint idx = 0; idx < N; ++idx) {
    uint x = idx;
    for (uint i = m.n; i-- > 0; x /= m.D)
      path[i] = x % m.D;
    double logw = 0.0;
    for (uint i = 0; i < m.n && logw > -INF; ++i) {
      double u = m.U(path, i, path[i]);
      logw = (u == INF) ? -INF : logw + std::log(m.b[path[i]]) - u;
    }
    z += std::exp(logw);
  }
  EXPECT_NEAR(ExactLogZ(m), std::log(z), 1e-10);
}

TEST(SMCGrowthTest, EstimatorIsUnbiased) {
  // E[Z_hat] = Z: the mean of Z_hat / Z over many runs must be 1.
  Model m = RandomModel(8, 6, 1.0, 0.3, 0.1, 11);
  double logZ = ExactLogZ(m);
  std::mt19937_64 gen(5);
  Uniform uni(gen);
  const double cases[][3] = {{1, 3, 2.0}, {6, 2, 2.0}, {6, 2, 0.5}, {6, 2, 0.0}};
  for (const auto &c : cases) {
    SMC smc((uint)c[0], c[2]);
    uint k = (uint)c[1];
    const uint R = 200000;
    double s = 0.0, s2 = 0.0;
    for (uint r = 0; r < R; ++r) {
      double x = std::exp(smc.GrowNew(ToyParticle(m, uni, k), uni) - logZ);
      s += x;
      s2 += x * x;
    }
    double mean = s / R, se = std::sqrt((s2 / R - mean * mean) / R);
    EXPECT_LT(std::fabs(mean - 1.0), 5.0 * se)
        << "N=" << c[0] << " k=" << c[1] << " tau=" << c[2];
    std::cout << "[ SMC     ] N=" << c[0] << " k=" << c[1] << " tau=" << c[2]
              << "  E[Z_hat/Z] " << mean << " +- " << se << "\n";
  }
}

TEST(SMCGrowthTest, MatchesExactDistribution) {
  Model m = SmallModel();
  // N, k, tau (2 = resample every step, 0 = never)
  const double cases[][3] = {{1, 1, 2.0}, {1, 3, 2.0}, {2, 1, 2.0},
                             {4, 1, 2.0}, {4, 2, 2.0}, {3, 2, 0.0},
                             {5, 2, 0.5}, {8, 1, 2.0}};
  for (const auto &c : cases) {
    uint N = c[0], k = c[1];
    ChainResult r = RunChain(m, N, k, c[2], 1000000, 31 + N * 10 + k);
    EXPECT_LT(r.maxZ, 5.5) << "N=" << N << " k=" << k << " tau=" << c[2];
    std::cout << "[ SMC     ] N=" << N << " k=" << k << " tau=" << c[2]
              << "  acceptance " << r.acc << "  TV " << r.tv << "  max|z| "
              << r.maxZ << "\n";
  }
}

TEST(SMCGrowthTest, UnconditionalOldWeightIsDetected) {
  // Weighing the old configuration with a fresh unconditional run is the
  // obvious mistake; the exactness check must catch it.
  Model m = SmallModel();
  ChainResult r = RunChain(m, 4, 2, 2.0, 1000000, 99, true);
  std::cout << "[ SMC     ] broken old weight: max|z| " << r.maxZ << "  TV "
            << r.tv << "\n";
  EXPECT_GT(r.maxZ, 8.0);
}

TEST(SMCGrowthTest, ZeroWeightOldIsGrownInFull) {
  // An old configuration with infinite energy (an overlapping start). Its
  // step weight is the mean over its trial set -- the actual placement plus
  // k - 1 fresh trials -- so with k = 1 it is 0 and WeighOld must return
  // Z_hat_old = 0, as single-chain CBMC gives W_old = 0. Either way it must
  // hand back the whole old configuration: GOMC takes the old molecule's
  // energy from it, and a partial one left the running energy wrong when
  // the molecule was removed.
  Model m = SmallModel();
  const std::vector<int> actual = {0, 2, 0, 1, 3}; // e1[0][2] = INF at step 1
  ASSERT_EQ(m.U(actual, 1, actual[1]), INF);
  std::mt19937_64 gen(5);
  Uniform uni(gen);
  for (uint N : {1u, 2u, 16u})
    for (uint k : {1u, 4u}) {
      SMC smc(N, 2.0);
      double lo = smc.WeighOld(ToyParticle(m, uni, k, &actual), uni);
      if (k == 1)
        EXPECT_EQ(lo, -INF) << "N=" << N;
      EXPECT_EQ(smc.Chosen().path, actual) << "N=" << N << " k=" << k;
    }
}

// ---------------------------------------------------------------------------
// Benchmark: equal trials per step (N * k), longer random chains.
// Disabled by default (about 7 min). Run with
//   --gtest_also_run_disabled_tests --gtest_filter=SMCGrowthBench.*
namespace {
struct BenchRow {
  double mean, sd, fail, acc;
};

BenchRow Bench(const Model &m, double logZ, uint N, uint k, double tau,
               uint runs, uint moves, uint seed) {
  std::mt19937_64 gen(seed);
  Uniform uni(gen);
  SMC smc(N, tau);
  double s = 0.0, s2 = 0.0;
  uint ok = 0;
  for (uint r = 0; r < runs; ++r) {
    double l = smc.GrowNew(ToyParticle(m, uni, k), uni);
    if (l == -INF)
      continue;
    ++ok;
    s += l - logZ;
    s2 += (l - logZ) * (l - logZ);
  }
  BenchRow row;
  row.fail = 1.0 - (double)ok / runs;
  row.mean = ok ? s / ok : NAN;
  row.sd = ok > 1 ? std::sqrt(std::max(0.0, s2 / ok - row.mean * row.mean)) : NAN;
  // Markov chain acceptance, starting from a long-run sample.
  SMC start(256, 2.0);
  std::vector<int> cur;
  if (start.GrowNew(ToyParticle(m, uni, 4), uni) > -INF)
    cur = start.Chosen().path;
  uint acc = 0;
  for (uint it = 0; it < moves && !cur.empty(); ++it) {
    double ln = smc.GrowNew(ToyParticle(m, uni, k), uni);
    std::vector<int> trial = smc.Chosen().path;
    double lo = smc.WeighOld(ToyParticle(m, uni, k, &cur), uni);
    if (ln > -INF && std::log(uni()) < ln - lo) {
      cur = trial;
      ++acc;
    }
  }
  row.acc = (double)acc / moves;
  return row;
}
} // namespace

TEST(SMCGrowthBench, DISABLED_EqualCost) {
  struct Sys {
    const char *name;
    uint n, D;
    double sigma, closed, closed0;
  };
  // difficulty scan: chain length and fraction of forbidden pairs
  const Sys systems[] = {{"n=16 D=8  s=1.0 c=0.20/0.10", 16, 8, 1.0, 0.2, 0.1},
                         {"n=30 D=12 s=1.0 c=0.20/0.10", 30, 12, 1.0, 0.2, 0.1},
                         {"n=30 D=12 s=1.0 c=0.30/0.20", 30, 12, 1.0, 0.3, 0.2},
                         {"n=30 D=12 s=1.0 c=0.40/0.20", 30, 12, 1.0, 0.4, 0.2},
                         {"n=60 D=12 s=1.0 c=0.20/0.10", 60, 12, 1.0, 0.2, 0.1},
                         {"n=60 D=12 s=1.0 c=0.30/0.20", 60, 12, 1.0, 0.3, 0.2}};
  const uint budgets[] = {32, 128};
  for (const Sys &sy : systems) {
    Model m = RandomModel(sy.n, sy.D, sy.sigma, sy.closed, sy.closed0, 7);
    double logZ = ExactLogZ(m);
    for (uint B : budgets) {
      // (N, k, tau) with N * k = B trials per step
      const double methods[][3] = {
          {1, (double)B, 2.0},     {B / 4.0, 4, 0.0}, {(double)B, 1, 2.0},
          {B / 2.0, 2, 2.0},       {B / 4.0, 4, 2.0}, {B / 8.0, 8, 2.0},
          {B / 4.0, 4, 0.5}};
      std::cout << "\n== " << sy.name << "  log Z = " << logZ
                << "  trials/step = " << B << "\n"
                << "   N   k   tau | mean(lnZh-lnZ)  sd(lnZh)  P(fail)  "
                   "MC acceptance\n";
      for (const auto &c : methods) {
        BenchRow r = Bench(m, logZ, (uint)c[0], (uint)c[1], c[2], 2000, 10000,
                           123);
        std::cout << std::fixed << std::setprecision(3) << std::setw(4)
                  << (uint)c[0] << std::setw(4) << (uint)c[1] << std::setw(6)
                  << c[2] << " | " << std::setw(13) << r.mean << std::setw(10)
                  << r.sd << std::setw(9) << r.fail << std::setw(12) << r.acc
                  << "\n";
      }
    }
  }
}
