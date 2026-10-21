/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#ifndef RECOILGROWTH_H
#define RECOILGROWTH_H

#include <algorithm>
#include <vector>

#include "BasicTypes.h"

namespace cbmc {

// Recoil growth (S. Consta, N. B. Wilding, D. Frenkel, Z. Alexandrowicz,
// J. Chem. Phys. 110, 3220 (1999)) over a fixed sequence of growth steps.
//
// Each step draws k trials from the provider. Trial t is open with
// probability p(t), one coin flip per trial. A trial is feasible at depth r
// if it is open and, unless r == 1 or it is the last step, at least one of
// the k trials generated beneath it is feasible at depth r - 1. Growth picks
// among the trials feasible at depth l (the recoil length) with probability
// proportional to their weight w and moves on; it fails if a step has none.
// With S_i the summed weight of the feasible trials at step i, the
// configuration's weight is
//
//     W = prod_i  S_i / (k * p(t_i)),
//
// where w is the trial's full Rosenbluth weight (bonded importance weight
// times exp(-beta u)). For any p in (0, 1] that depends only on the trial and
// the atoms already placed, acc = min(1, W_new / W_old) satisfies detailed
// balance (super-detailed balance over the tree of trials and coin flips).
// The old configuration's actual placement is trial 0 at every step and is
// feasible by construction; its other k - 1 trials are generated and tested
// the same way.
//
// The original method picks uniformly, W = prod_i m_i w(t_i) / (k p(t_i)),
// with p = min(1, exp(-beta u)), which suits hard-core potentials. Weighted
// selection reduces to standard CBMC when every trial is open, so recoil
// growth then only removes dead-end trials from the choice.
//
// Trial sets are generated lazily but once: a node's trials and coin flips
// are kept for the rest of the growth, so this is equivalent to searching a
// tree drawn in full beforehand.
//
// Provider interface:
//   typedef ... Trial;          // must have double fields w and p
//   uint NumSteps() const;
//   void Generate(uint step, uint n, Trial *out); // n trials given the atoms
//                                                 // placed so far
//   void Actual(uint step, Trial &out);           // old configuration only
//   void Place(uint step, const Trial &t);        // tentative placement
//   void Unplace(uint step);                      // undo the last Place
// Rng: double operator()() uniform in [0, 1).
template <class Provider, class Rng> class RecoilGrowth {
public:
  typedef typename Provider::Trial Trial;

  RecoilGrowth(uint trials, uint length)
      : k(trials < 1 ? 1 : trials), l(length < 1 ? 1 : length) {}

  // Grow a new configuration. Returns W (0 if growth failed). On success the
  // chosen trials are left placed and are available through Chosen().
  double GrowNew(Provider &prov, Rng &rng);
  // Weigh the old configuration, whose actual placements the provider
  // returns through Actual(). Leaves the actual trials placed.
  double WeighOld(Provider &prov, Rng &rng);

  // Keep each step's trial set for diagnostics (StepTrials/StepOpen).
  void KeepSteps(bool keep) { keepSteps = keep; }
  uint Trials() const { return k; }
  uint Length() const { return l; }
  // Results of the last GrowNew/WeighOld, per step.
  uint StepsDone() const { return (uint)factor.size(); }
  const Trial &Chosen(uint step) const { return chosen[step]; }
  double Factor(uint step) const { return factor[step]; }
  uint FeasibleCount(uint step) const { return feasible[step]; }
  // The trial set the choice at `step` was made from.
  const std::vector<Trial> &StepTrials(uint step) const {
    return stepTrials[step];
  }
  const std::vector<char> &StepOpen(uint step) const { return stepOpen[step]; }

private:
  struct Node {
    uint step;
    std::vector<Trial> t;
    std::vector<char> open;
    std::vector<int> child;          // node index, -1 if not generated
    std::vector<unsigned char> ok;   // largest depth known feasible
    std::vector<unsigned char> bad;  // smallest depth known infeasible
  };

  int NewNode(uint step) {
    if (used == pool.size())
      pool.push_back(Node());
    Node &nd = pool[used];
    nd.step = step;
    nd.t.resize(k);
    nd.open.assign(k, 0);
    nd.child.assign(k, -1);
    nd.ok.assign(k, 0);
    nd.bad.assign(k, 255);
    return (int)used++;
  }

  void Flip(int ni, uint first, Rng &rng) {
    for (uint j = first; j < k; ++j) {
      double p = pool[ni].t[j].p;
      pool[ni].open[j] = (p >= 1.0) || (p > 0.0 && rng() < p);
    }
  }

  int GenerateNode(uint step, Provider &prov, Rng &rng) {
    int ni = NewNode(step);
    prov.Generate(step, k, pool[ni].t.data());
    Flip(ni, 0, rng);
    return ni;
  }

  int Child(int ni, uint j, Provider &prov, Rng &rng) {
    int c = pool[ni].child[j];
    if (c < 0) {
      c = GenerateNode(pool[ni].step + 1, prov, rng);
      pool[ni].child[j] = c;
    }
    return c;
  }

  // Feasibility of trial j of node ni at depth r. The path to node ni must
  // be placed; trial j is placed while its subtree is searched.
  bool Feasible(int ni, uint j, uint r, Provider &prov, Rng &rng) {
    if (!pool[ni].open[j])
      return false;
    uint step = pool[ni].step;
    if (r <= 1 || step + 1 >= nSteps)
      return true;
    if (pool[ni].ok[j] >= r)
      return true;
    if (pool[ni].bad[j] <= r)
      return false;
    prov.Place(step, pool[ni].t[j]);
    int c = Child(ni, j, prov, rng);
    bool f = false;
    for (uint cj = 0; cj < k && !f; ++cj)
      f = Feasible(c, cj, r - 1, prov, rng);
    prov.Unplace(step);
    unsigned char rc = (unsigned char)std::min(r, 254u);
    if (f)
      pool[ni].ok[j] = std::max(pool[ni].ok[j], rc);
    else
      pool[ni].bad[j] = std::min(pool[ni].bad[j], rc);
    return f;
  }

  void Record(int ni, uint m, double f, const Trial &t) {
    feasible.push_back(m);
    factor.push_back(f);
    chosen.push_back(t);
    if (keepSteps) {
      stepTrials.push_back(pool[ni].t);
      stepOpen.push_back(pool[ni].open);
    }
  }

  void Reset(Provider &prov) {
    nSteps = prov.NumSteps();
    used = 0;
    chosen.clear();
    factor.clear();
    feasible.clear();
    stepTrials.clear();
    stepOpen.clear();
  }

  uint k, l, nSteps = 0;
  bool keepSteps = false;
  std::vector<Node> pool;
  size_t used = 0;
  std::vector<Trial> chosen;
  std::vector<double> factor;
  std::vector<uint> feasible;
  std::vector<std::vector<Trial>> stepTrials;
  std::vector<std::vector<char>> stepOpen;
};

template <class Provider, class Rng>
double RecoilGrowth<Provider, Rng>::GrowNew(Provider &prov, Rng &rng) {
  Reset(prov);
  if (nSteps == 0)
    return 1.0;
  int cur = GenerateNode(0, prov, rng);
  double W = 1.0;
  std::vector<uint> feas;
  for (uint i = 0; i < nSteps; ++i) {
    feas.clear();
    for (uint j = 0; j < k; ++j)
      if (Feasible(cur, j, l, prov, rng))
        feas.push_back(j);
    double sum = 0.0;
    for (uint j : feas)
      sum += pool[cur].t[j].w;
    if (!(sum > 0.0)) {
      Record(cur, (uint)feas.size(), 0.0, pool[cur].t[0]);
      for (uint s = i; s-- > 0;)
        prov.Unplace(s);
      return 0.0;
    }
    // pick a feasible trial with probability w / sum
    double x = rng() * sum, c = 0.0;
    uint j = feas.back();
    for (uint f : feas) {
      c += pool[cur].t[f].w;
      if (x < c) {
        j = f;
        break;
      }
    }
    const Trial t = pool[cur].t[j];
    double f = sum / (k * t.p);
    Record(cur, (uint)feas.size(), f, t);
    W *= f;
    prov.Place(i, t);
    if (i + 1 < nSteps)
      cur = Child(cur, j, prov, rng);
  }
  return W;
}

template <class Provider, class Rng>
double RecoilGrowth<Provider, Rng>::WeighOld(Provider &prov, Rng &rng) {
  Reset(prov);
  double W = 1.0;
  for (uint i = 0; i < nSteps; ++i) {
    // nodes of earlier steps are no longer needed
    used = 0;
    int ni = NewNode(i);
    prov.Actual(i, pool[ni].t[0]);
    pool[ni].open[0] = 1;
    if (k > 1)
      prov.Generate(i, k - 1, &pool[ni].t[1]);
    Flip(ni, 1, rng);
    uint m = 1;
    double sum = pool[ni].t[0].w;
    for (uint j = 1; j < k; ++j)
      if (Feasible(ni, j, l, prov, rng)) {
        ++m;
        sum += pool[ni].t[j].w;
      }
    const Trial t = pool[ni].t[0];
    // the actual trial is open by construction; guard p against underflow
    double f = sum / (k * std::max(t.p, 1e-300));
    Record(ni, m, f, t);
    W *= f;
    prov.Place(i, t);
  }
  return W;
}

} // namespace cbmc

#endif /*RECOILGROWTH_H*/
