/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#ifndef FLEXIBLEBOND_H
#define FLEXIBLEBOND_H

#include <cmath>

#include "BasicTypes.h"
#include "FFBonds.h"
#include "MolSetup.h"

namespace cbmc {

// Flexible bonds in CBMC growth (FlexibleBonds keyword).
//
// A flexible bond's length is drawn exactly from its ideal distribution
//     p(l) ~ l^2 exp(-beta Kb (l - b0)^2),  l > 0,
// where l^2 is the Jacobian of placing an atom at distance l from its focus
// (the direction is sampled separately, as for fixed bonds). Because the draw
// is exact, its contribution to the Rosenbluth weight is the normalisation
// constant of p, the same for every configuration of the molecule kind, so it
// cancels in every CBMC acceptance ratio (in GCMC it shifts the ideal-gas
// reference of the chemical potential by a constant, as the bend and torsion
// normalisations already do). The bond energy is therefore added to the
// molecule's energy but not to its weight, as for fixed bonds.
//
// Bonds to hydrogen, bonds with a fixed (or non-positive) Kb, and ring bonds
// (DCCyclic) are not sampled.

// Atoms lighter than this (H, D, T and massless sites) keep their bonds fixed.
static const double FLEXIBLE_BOND_MIN_MASS = 3.5;

// Is the bond a0-a1 of bond kind `bondKind` sampled?
inline bool IsFlexibleBond(bool enabled, const FFBonds &bonds,
                           const mol_setup::MolKind &kind, uint a0, uint a1,
                           uint bondKind) {
  return enabled && !bonds.Fixed(bondKind) && bonds.Stiffness(bondKind) > 0.0 &&
         kind.atoms[a0].mass >= FLEXIBLE_BOND_MIN_MASS &&
         kind.atoms[a1].mass >= FLEXIBLE_BOND_MIN_MASS;
}
template <class Data>
inline bool IsFlexibleBond(const Data *data, const mol_setup::MolKind &kind,
                           uint a0, uint a1, uint bondKind) {
  return IsFlexibleBond(data->flexibleBonds, data->ff.bonds, kind, a0, a1,
                        bondKind);
}

// Exact draw from p(l) ~ l^2 exp(-(l - b0)^2 / (2 s^2)), s^2 = 1/(2 beta Kb).
// Rejection from a Gaussian envelope: with m the mode of p, ln l <= ln m +
// l/m - 1 gives l^2 <= m^2 exp(2l/m - 2), so p is bounded by a Gaussian of
// width s centred on b0 + 2 s^2 / m; a draw is kept with probability
// ((l/m) exp(1 - l/m))^2 <= 1. The envelope touches p at its mode, so nearly
// every draw is kept (acceptance > 99.9% for C-C at room temperature).
// rng() is uniform in [0, 1), rng.Gaussian(mean, sd) normal.
template <class Rng>
inline double SampleBondLength(Rng &rng, double b0, double kb, double beta) {
  const double s2 = 0.5 / (beta * kb);
  const double s = std::sqrt(s2);
  const double m = 0.5 * (b0 + std::sqrt(b0 * b0 + 8.0 * s2));
  const double mu = b0 + 2.0 * s2 / m;
  for (;;) {
    double l = rng.Gaussian(mu, s);
    if (l <= 0.0)
      continue;
    double x = l / m;
    double a = x * std::exp(1.0 - x);
    if (rng() < a * a)
      return l;
  }
}

} // namespace cbmc

#endif /*FLEXIBLEBOND_H*/
