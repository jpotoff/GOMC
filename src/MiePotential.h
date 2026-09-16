/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) 2.80
Copyright (C) 2022 GOMC Group
A copy of the MIT License can be found in License.txt
along with this program, also can be found at
<https://opensource.org/licenses/MIT>.
********************************************************************************/
#ifndef MIE_POTENTIAL_H
#define MIE_POTENTIAL_H

#include "BasicTypes.h"
#include "NumLib.h"
#include <algorithm>
#include <cmath>

//
// The Mie repulsive/attractive terms, in one place.
//
// This block was previously duplicated at 13 sites across FFParticle, FF_SHIFT,
// FF_SWITCH and FF_SWITCH_MARTINI -- the ordinary and 1-4 paths of each, plus
// the virial forms. That duplication is why a wrong exponent had to be fixed in
// eleven places, and why the GPU kernels still carry an older version of it.
//
// The four forcefields differ only in what they do with these two terms, not in
// how the terms are computed, so they are computed here once.
//
// GOMC_HOSTDEV is a no-op today. It marks the functions that are candidates for
// sharing with the CUDA kernels, which currently reimplement this arithmetic.
//
#ifndef GOMC_HOSTDEV
#ifdef __CUDACC__
#define GOMC_HOSTDEV __host__ __device__
#else
#define GOMC_HOSTDEV
#endif
#endif

namespace ff {

// Sentinel stored in nExp when the exponent is not a usable integer, meaning
// the std::pow fallback must be taken.
const uint MIE_EXP_NOT_INTEGER = 0xFFFFFFFF;

// Range of integer exponents the fast path is valid for. Both bounds are load
// bearing and are used by FFParticle::Blend when it fills nExp:
//   lower -- the Mie form requires n > 6 (the prefactor n/(n-6) diverges at 6)
//   upper -- num::POW's switch table runs to case 25, i.e. it handles e <= 51
// Raising MIE_EXP_MAX past what num::POW tabulates would silently return a
// wrong repulsive term rather than fail, so the two must be changed together.
const uint MIE_EXP_MIN = 7;
const uint MIE_EXP_MAX = 50;

struct MieTerms {
  double attract; //!< (sigma/r)^6
  double repulse; //!< (sigma/r)^n
};

//
// Both terms of the Mie potential from rRat2 = (sigma/r)^2.
//
// `nExp` is the precomputed integer exponent (MIE_EXP_NOT_INTEGER if n is not a
// usable integer) and `n` the exponent as read from the parameter file.
//
// The operation order here is deliberately identical to the code this replaces,
// so results are bit-for-bit unchanged.
//
GOMC_HOSTDEV inline MieTerms MiePair(const double rRat2, const uint nExp,
                                     const double n) {
  const double rRat4 = rRat2 * rRat2;
  const double attract = rRat4 * rRat2;

  double repulse;
  if (nExp == 12) {
    // n = 12 is overwhelmingly the common case, and (sigma/r)^12 is just the
    // attractive term squared -- no table lookup, no pow.
    repulse = attract * attract;
  } else if (nExp != MIE_EXP_NOT_INTEGER) {
    repulse = num::POW(rRat2, rRat4, attract, nExp);
  } else {
    repulse = pow(rRat2, n * 0.5);
  }
  return {attract, repulse};
}

//
// Soft-core free-energy path.
//
// When a molecule is being coupled in or out (lambda < 1) the pair is evaluated
// at a softened separation instead of the real one, so the potential stays
// finite as the molecule is annihilated. This block was previously duplicated
// at 40 sites -- every CalcEn/CalcVir/CalcCoulomb/CalcCoulombVir/CalcdEndL/
// CalcCoulombdEndL of all five forcefields -- and is identical in all of them.
//
struct SoftCoreDist {
  double softRsq; //!< softened r^2, to evaluate the ordinary potential at
  double sigma6;  //!< the clamped sigma^6 used; dE/dlambda needs it again
};

//
// Operation order matches the code this replaces, so results are unchanged.
//
GOMC_HOSTDEV inline SoftCoreDist
SoftenedDistance(const double distSq, const double sigmaSqIJ,
                 const double lambda, const double scAlpha, const uint scPower,
                 const double scSigma6) {
  double sigma6 = sigmaSqIJ * sigmaSqIJ * sigmaSqIJ;
  sigma6 = std::max(sigma6, scSigma6);
  const double dist6 = distSq * distSq * distSq;
  const double lambdaCoef = scAlpha * pow((1.0 - lambda), scPower);
  const double softDist6 = lambdaCoef * sigma6 + dist6;
  return {cbrt(softDist6), sigma6};
}

} // namespace ff

#endif /*MIE_POTENTIAL_H*/
