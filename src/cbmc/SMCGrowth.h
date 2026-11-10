/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#ifndef SMCGROWTH_H
#define SMCGROWTH_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "BasicTypes.h"

namespace cbmc {

// Population chain growth: sequential Monte Carlo (SMC) over the CBMC growth
// steps, with conditional SMC for the old configuration.
//
// N particles (partial chains) grow side by side. At step i each particle
// draws k trials and picks one with probability proportional to its weight w
// (exactly as single-chain CBMC does; see Particle::Extend). Its step weight
// is
//     W_i^n = (1/k) sum_j w_ij^n,
// and its running log weight L^n accumulates log W_i^n. When the effective
// sample size of exp(L) falls below tau * N (tau > 1: every step, tau = 0:
// never), the population is resampled multinomially in proportion to exp(L),
// the normaliser mean_n exp(L^n) is folded into the estimate and L is reset.
// After the last step the normaliser is folded in once more and one particle
// is chosen in proportion to exp(L). The estimate
//     Z_hat = prod over resampling epochs of mean_n exp(L^n)
// is an unbiased estimator of the Rosenbluth normalisation (Del Moral 2004).
// N = 1 is standard CBMC with k trials; tau = 0 is N independent CBMC chains
// whose weights are averaged.
//
// Exactness. Accepting with min(1, Z_hat_new / Z_hat_old) samples the target
// exactly when Z_hat_old comes from *conditional* SMC (Andrieu, Doucet &
// Holenstein, J. R. Stat. Soc. B 72, 269 (2010)): particle 0 is the old
// configuration, its trial set at every step is its actual placement plus
// k - 1 fresh trials, it always survives resampling as its own ancestor, and
// the other N - 1 particles are grown and resampled (from all N, including
// particle 0) as usual. The move is then a Gibbs refresh of the auxiliary
// particles given the current configuration followed by a particle
// independent Metropolis-Hastings step. Weighing the old configuration with
// an unconditional run breaks detailed balance.
//
// Weights are kept as logs so long chains do not underflow; the returned
// value is log Z_hat (-infinity if every particle died at some step).
//
// Particle interface (copied on resampling, so it must be cheap to copy):
//   uint NumSteps() const;
//   // Grow one step: draw this particle's trials, place one, and return
//   // the step weight (the mean trial weight; 0 if every trial is dead).
//   // With reference = true the particle is the old configuration: its
//   // actual placement is one of its trials and is the one placed.
//   double Extend(uint step, bool reference);
// The particle owns the per-step trial logic (standard CBMC with k trials in
// the test and in GOMC); the engine owns only weights and resampling.
// Rng: double operator()() uniform in [0, 1).
template <class Particle, class Rng> class SMCGrowth {
public:
  SMCGrowth(uint particles, double essThreshold = 2.0)
      : N(particles < 1 ? 1 : particles), tau(essThreshold) {}

  // Grow a new configuration from the empty particle `proto`. Returns
  // log Z_hat; on success the chosen particle is available through Chosen().
  double GrowNew(const Particle &proto, Rng &rng) {
    return Run(proto, rng, false);
  }
  // Weigh the old configuration, which `proto` returns through Actual().
  // Particle 0 (Chosen() afterwards) carries the actual placements.
  double WeighOld(const Particle &proto, Rng &rng) {
    return Run(proto, rng, true);
  }

  const Particle &Chosen() const { return parts[chosen]; }
  uint Particles() const { return N; }
  // Diagnostics for the last run.
  uint Resamplings() const { return nResample; }
  uint DeadSteps() const { return nDeadSteps; } // particle-steps, all k dead

private:
  static double NegInf() { return -std::numeric_limits<double>::infinity(); }

  // log mean_n exp(L[n]); -inf if all are -inf
  double LogMeanExp(const std::vector<double> &L) const {
    double mx = NegInf();
    for (double l : L)
      if (l > mx)
        mx = l;
    if (mx == NegInf())
      return mx;
    double s = 0.0;
    for (double l : L)
      s += std::exp(l - mx);
    return mx + std::log(s / L.size());
  }

  // normalised weights exp(L) into p; returns the ESS
  double Normalise(const std::vector<double> &L, std::vector<double> &p) const {
    double mx = NegInf();
    for (double l : L)
      if (l > mx)
        mx = l;
    p.resize(L.size());
    double s = 0.0, s2 = 0.0;
    for (size_t n = 0; n < L.size(); ++n) {
      p[n] = (L[n] == NegInf()) ? 0.0 : std::exp(L[n] - mx);
      s += p[n];
    }
    for (double &x : p) {
      x /= s;
      s2 += x * x;
    }
    return 1.0 / s2;
  }

  uint Draw(const std::vector<double> &cum, Rng &rng) const {
    double x = rng() * cum.back();
    uint lo = 0, hi = (uint)cum.size() - 1;
    while (lo < hi) {
      uint mid = (lo + hi) / 2;
      if (x < cum[mid])
        hi = mid;
      else
        lo = mid + 1;
    }
    return lo;
  }

  double Run(const Particle &proto, Rng &rng, bool conditional) {
    const uint nSteps = proto.NumSteps();
    parts.assign(N, proto);
    L.assign(N, 0.0);
    nResample = nDeadSteps = 0;
    double logZ = 0.0;

    for (uint step = 0; step < nSteps; ++step) {
      for (uint n = 0; n < N; ++n) {
        if (L[n] == NegInf())
          continue; // dead particle: never resampled again until reset
        double W = parts[n].Extend(step, conditional && n == 0);
        if (!(W > 0.0)) {
          L[n] = NegInf();
          ++nDeadSteps;
          continue;
        }
        L[n] += std::log(W);
      }

      bool last = (step + 1 == nSteps);
      double ess = Normalise(L, p);
      bool allDead = true;
      for (double l : L)
        allDead = allDead && (l == NegInf());
      if (allDead)
        return NegInf(); // cannot happen when conditional (particle 0 lives)
      if (last || ess < tau * N) {
        logZ += LogMeanExp(L);
        cum.resize(N);
        double c = 0.0;
        for (uint n = 0; n < N; ++n)
          cum[n] = (c += p[n]);
        if (last) {
          chosen = conditional ? 0 : Draw(cum, rng);
          break;
        }
        ++nResample;
        // with conditional SMC, particle 0 keeps its own lineage
        uint first = conditional ? 1 : 0;
        anc.resize(N);
        for (uint n = 0; n < first; ++n)
          anc[n] = n;
        for (uint n = first; n < N; ++n)
          anc[n] = Draw(cum, rng);
        // Particles are exchangeable, so their order is free: sorting puts
        // the copies of each parent next to each other, which lets the
        // particle reuse per-parent work (GOMC's neighbour candidate list).
        std::sort(anc.begin() + first, anc.end());
        next = parts; // particles need not be default-constructible
        for (uint n = first; n < N; ++n)
          if (anc[n] != n)
            next[n] = parts[anc[n]];
        parts.swap(next);
        L.assign(N, 0.0);
      }
    }
    return logZ;
  }

  uint N;
  double tau;
  std::vector<Particle> parts, next;
  std::vector<double> L, p, cum;
  std::vector<uint> anc;
  uint chosen = 0, nResample = 0, nDeadSteps = 0;
};

} // namespace cbmc

#endif /*SMCGROWTH_H*/
