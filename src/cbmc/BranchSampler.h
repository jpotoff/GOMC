/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#ifndef BRANCHSAMPLER_H
#define BRANCHSAMPLER_H

#include <algorithm>
#include <cmath>
#include <vector>

#include "BasicTypes.h"
#include "CBMC.h"
#include "FFAngles.h"

namespace cbmc {

// Tabulated envelope for sampling a bond angle theta from
// p(theta) ~ sin(theta) exp(-beta (U(theta) - uMin)) by bin rejection.
// pMax[i] bounds p over bin i; cumMax is its running sum.
struct BendTable {
  double beta = 0.0; // beta the table was built for (0 = not built)
  double uMin = 0.0; // minimum of U(theta) on [0, pi]
  double width = 0.0;
  std::vector<double> pMax;
  std::vector<double> cumMax;
};

// Generates trial geometries for the atoms bonded to a focus atom (the
// branch), given the previously placed atom prev, for any number of branches
// and any bond angle geometry. The polar axis is focus->prev; theta[b] is the
// prev-focus-b angle and phi[b] the twist of b about the axis (phi[0] = 0).
// The target density is
//   p ~ prod_b sin(theta_b) exp(-beta U(prev,focus,b))
//       * prod_{c<b} exp(-beta U(c,focus,b)).
//
// This is a generalization of the Jacobian-Gaussian scheme (Sepehri,
// Loeffler, Chen, J. Chem. Theory Comput. 13, 1577 (2017)):
//  - theta_b is drawn exactly from sin(theta) exp(-beta U) with a table, which
//    covers any angle potential and near-linear angles;
//  - for b >= 1, the angle to one reference atom c* (the stiffest free pair)
//    is drawn from a truncated Gaussian fit to its bending potential, and
//    phi_b follows from the spherical law of cosines. Of the two mirror
//    solutions, one is chosen with probability proportional to the Boltzmann
//    factor of the remaining pair angles at each, which avoids proposing an
//    atom on top of one already placed. A small uniform-phi component is
//    mixed in so the weights stay bounded near planar configurations, where
//    the Jacobian is singular;
//  - all other pair angles are enforced by exact Boltzmann rejection.
// Centers whose equilibrium geometry is planar or linear about the reference
// pair use uniform phi with rejection instead (the paper's (theta, omega)
// variant). Twist angles constrained by fixed bond angles are solved for.
//
// Sample() returns a trial; Weight() is its importance weight relative to
// the target, up to a constant fixed for the branch. Callers select among N
// trials with probability proportional to the weights and use the summed
// weight as the Rosenbluth factor (the old configuration's weight comes from
// Weight() on the actual geometry), which keeps the move exact.
class BranchSampler {
public:
  void Init(uint nBonds, const uint angleKinds[MAX_BONDS][MAX_BONDS],
            const FFAngles *angles, std::vector<BendTable> *tables);

  // Draws one trial geometry. Returns false if no geometry satisfying the
  // rejection tests was found, which indicates inconsistent angle parameters.
  template <class RNG>
  bool Sample(RNG &rng, double beta, double *th, double *ph);

  // Importance weight of a branch geometry (new trial or old configuration).
  double Weight(double beta, const double *th, const double *ph);

  // Angle between bonded atoms b and c from their spherical coordinates
  static double AngleBetween(double tb, double pb, double tc, double pc) {
    double x = sin(tb) * sin(tc) * cos(pb - pc) + cos(tb) * cos(tc);
    return acos(std::max(-1.0, std::min(1.0, x)));
  }

  // Fixed bond angle constraints for branch atom b could not be satisfied
  int FailedFixedPair() const { return failedFixed; }

private:
  const BendTable &Table(uint kind, double beta);
  template <class RNG> double SampleTheta(RNG &rng, uint kind, double beta);
  template <class RNG> double Normal(RNG &rng);
  template <class RNG>
  double TruncNormal(RNG &rng, double mu, double sigma, double lo, double hi,
                     double z);
  bool UseGaussian(uint b, double beta) const;
  // Mixing fraction of uniform phi for atom b given the sampled thetas, and
  // the support and normalization of the truncated Gaussian on theta(b,c*)
  double MixFraction(uint b, double beta, const double *th, double &lo,
                     double &hi, double &z) const;
  double PhiDensity(uint b, double beta, const double *th, const double *ph);
  // Boltzmann factor of the pair angles of atom b at twist phi, excluding
  // fixed angles and (if skipRef) the reference pair
  double RestFactor(uint b, double beta, const double *th, const double *ph,
                    double phi, bool skipRef);
  bool SolveFixedPhi(uint b, const double *th, double *ph);

  uint nBonds = 0;
  uint kinds[MAX_BONDS][MAX_BONDS];
  const FFAngles *ang = nullptr;
  std::vector<BendTable> *tables = nullptr;
  std::vector<BendTable> ownTables;

  // Per branch atom b >= 1
  bool fixedPair[MAX_BONDS]; // b has a fixed bond angle to some c < b
  int ref[MAX_BONDS];        // reference atom c* for the Gaussian, or -1
  double refMu[MAX_BONDS];   // equilibrium angle of the (b, c*) pair
  double refCurv[MAX_BONDS]; // curvature of U(b, c*) at its minimum
  int failedFixed = -1;
};

namespace branch_detail {
const uint kBendBins = 1024;
const uint kBendSub = 8;          // sub-samples per bin for the bound
const double kBendSafety = 1.001; // margin on the per-bin bound
const unsigned long kMaxTries = 1000000;
const double kUniformMix = 0.05;  // uniform-phi fraction in the mixture
const double kMaxSigma = 0.6;     // widest Gaussian (rad) worth using
const double kPlanarCos = 0.95;   // |cos(omega0)| above this is planar
const double kMinSin = 0.1;       // sin(theta0) below this is near linear
const double kSignFloor = 1e-12;  // keeps both mirror solutions possible
} // namespace branch_detail

inline void BranchSampler::Init(uint n,
                                const uint angleKinds[MAX_BONDS][MAX_BONDS],
                                const FFAngles *angles,
                                std::vector<BendTable> *tbl) {
  using namespace branch_detail;
  nBonds = n;
  ang = angles;
  tables = tbl ? tbl : &ownTables;
  for (uint i = 0; i < n; ++i)
    for (uint j = 0; j < n; ++j)
      kinds[i][j] = angleKinds[i][j];

  for (uint b = 0; b < n; ++b) {
    fixedPair[b] = false;
    ref[b] = -1;
    double best = 0.0;
    for (uint c = 0; c < b; ++c) {
      uint k = kinds[b][c];
      if (ang->AngleFixed(k)) {
        fixedPair[b] = true;
        continue;
      }
      // equilibrium twist between b and c; skip planar or linear geometries
      double tb = ang->Angle(kinds[b][b]), tc = ang->Angle(kinds[c][c]);
      double t0 = ang->Angle(k);
      double s = sin(tb) * sin(tc);
      if (sin(tb) < kMinSin || sin(tc) < kMinSin)
        continue;
      double cosOmega0 = (cos(t0) - cos(tb) * cos(tc)) / s;
      if (std::abs(cosOmega0) > kPlanarCos)
        continue;
      const double h = 1e-4;
      double curv = (ang->Calc(k, t0 + h) - 2.0 * ang->Calc(k, t0) +
                     ang->Calc(k, t0 - h)) /
                    (h * h);
      if (curv > best) {
        best = curv;
        ref[b] = c;
        refMu[b] = t0;
        refCurv[b] = curv;
      }
    }
  }
}

// Build (or rebuild if beta changed) the sampling table for an angle kind.
inline const BendTable &BranchSampler::Table(uint kind, double beta) {
  using namespace branch_detail;
  if (tables->size() <= kind)
    tables->resize(kind + 1);
  BendTable &t = (*tables)[kind];
  if (t.beta == beta)
    return t;

  t.width = M_PI / kBendBins;
  // minimum of U on [0, pi], including theta0 where harmonic forms vanish
  t.uMin = ang->Calc(kind, ang->Angle(kind));
  for (uint i = 0; i <= kBendBins * kBendSub; ++i)
    t.uMin = std::min(t.uMin, ang->Calc(kind, i * t.width / kBendSub));

  t.pMax.assign(kBendBins, 0.0);
  t.cumMax.assign(kBendBins, 0.0);
  double sum = 0.0;
  for (uint i = 0; i < kBendBins; ++i) {
    double pm = 0.0;
    for (uint s = 0; s <= kBendSub; ++s) {
      double th = (i + double(s) / kBendSub) * t.width;
      pm = std::max(pm, sin(th) * exp(-beta * (ang->Calc(kind, th) - t.uMin)));
    }
    t.pMax[i] = pm * kBendSafety;
    sum += t.pMax[i];
    t.cumMax[i] = sum;
  }
  t.beta = beta;
  return t;
}

// Exact sample of theta from p(theta) ~ sin(theta) exp(-beta U(theta))
template <class RNG>
inline double BranchSampler::SampleTheta(RNG &rng, uint kind, double beta) {
  using namespace branch_detail;
  if (ang->AngleFixed(kind))
    return ang->Angle(kind);
  const BendTable &t = Table(kind, beta);
  while (true) {
    double r = rng() * t.cumMax.back();
    uint bin = std::upper_bound(t.cumMax.begin(), t.cumMax.end(), r) -
               t.cumMax.begin();
    bin = std::min(bin, kBendBins - 1);
    double th = (bin + rng()) * t.width;
    double p = sin(th) * exp(-beta * (ang->Calc(kind, th) - t.uMin));
    if (rng() * t.pMax[bin] < p)
      return th;
  }
}

template <class RNG> inline double BranchSampler::Normal(RNG &rng) {
  double u;
  do {
    u = rng();
  } while (u <= 1e-300);
  return sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * rng());
}

// Sample from a Gaussian truncated to [lo, hi] with normalization z
template <class RNG>
inline double BranchSampler::TruncNormal(RNG &rng, double mu, double sigma,
                                         double lo, double hi, double z) {
  if (z > 0.05) {
    for (uint i = 0; i < 1000; ++i) {
      double x = mu + sigma * Normal(rng);
      if (x >= lo && x <= hi)
        return x;
    }
  }
  // small acceptance window: uniform proposal on [lo, hi] with rejection
  double m = std::max(lo, std::min(hi, mu));
  while (true) {
    double x = lo + (hi - lo) * rng();
    double d = ((x - mu) * (x - mu) - (m - mu) * (m - mu)) / (2.0 * sigma * sigma);
    if (rng() < exp(-d))
      return x;
  }
}

inline bool BranchSampler::UseGaussian(uint b, double beta) const {
  return ref[b] >= 0 && !fixedPair[b] &&
         1.0 / sqrt(beta * refCurv[b]) < branch_detail::kMaxSigma;
}

inline double BranchSampler::MixFraction(uint b, double beta, const double *th,
                                         double &lo, double &hi,
                                         double &z) const {
  const uint c = ref[b];
  lo = std::abs(th[b] - th[c]);
  hi = std::min(th[b] + th[c], 2.0 * M_PI - th[b] - th[c]);
  z = 0.0;
  if (sin(th[b]) * sin(th[c]) < 1e-8 || hi <= lo)
    return 1.0;
  double sigma = 1.0 / sqrt(beta * refCurv[b]);
  double s2 = sigma * M_SQRT2;
  z = 0.5 * (erf((hi - refMu[b]) / s2) - erf((lo - refMu[b]) / s2));
  return z < 1e-8 ? 1.0 : branch_detail::kUniformMix;
}

inline double BranchSampler::RestFactor(uint b, double beta, const double *th,
                                        const double *ph, double phi,
                                        bool skipRef) {
  double en = 0.0;
  for (uint c = 0; c < b; ++c) {
    uint k = kinds[b][c];
    if (ang->AngleFixed(k) || (skipRef && (int)c == ref[b]))
      continue;
    en += ang->Calc(k, AngleBetween(th[b], phi, th[c], ph[c])) -
          Table(k, beta).uMin;
  }
  return exp(-beta * en);
}

// Proposal density of phi[b] given the thetas and the earlier twists
inline double BranchSampler::PhiDensity(uint b, double beta, const double *th,
                                        const double *ph) {
  using namespace branch_detail;
  double lo, hi, z;
  double mix = MixFraction(b, beta, th, lo, hi, z);
  double q = mix / (2.0 * M_PI);
  if (mix >= 1.0)
    return q;
  const uint c = ref[b];
  double d = remainder(ph[b] - ph[c], 2.0 * M_PI);
  double omega = std::abs(d);
  double x = AngleBetween(th[b], ph[b], th[c], ph[c]);
  double sx = sin(x);
  if (sx < 1e-12 || x < lo || x > hi)
    return q;
  double sigma = 1.0 / sqrt(beta * refCurv[b]);
  double gauss = exp(-0.5 * (x - refMu[b]) * (x - refMu[b]) / (sigma * sigma)) /
                 (sigma * sqrt(2.0 * M_PI) * z);
  // |d theta_bc / d phi_b|, from cos(theta_bc) = sin sin cos(omega) + cos cos
  double jac = sin(th[b]) * sin(th[c]) * sin(omega) / sx;
  // probability of having chosen this mirror solution
  double rThis = RestFactor(b, beta, th, ph, ph[b], true);
  double rOther = RestFactor(b, beta, th, ph, ph[c] - d, true);
  double pSign = (rThis + kSignFloor) / (rThis + rOther + 2.0 * kSignFloor);
  return q + (1.0 - mix) * gauss * jac * pSign;
}

template <class RNG>
inline bool BranchSampler::Sample(RNG &rng, double beta, double *th,
                                  double *ph) {
  using namespace branch_detail;
  failedFixed = -1;
  for (unsigned long tries = 0; tries < kMaxTries; ++tries) {
    bool accepted = true;
    for (uint b = 0; b < nBonds && accepted; ++b) {
      th[b] = SampleTheta(rng, kinds[b][b], beta);
      if (b == 0) {
        ph[0] = 0.0;
        continue;
      }
      bool gaussian = false;
      if (fixedPair[b]) {
        if (!SolveFixedPhi(b, th, ph))
          return false;
      } else if (UseGaussian(b, beta)) {
        gaussian = true;
        const uint c = ref[b];
        double lo, hi, z;
        double mix = MixFraction(b, beta, th, lo, hi, z);
        if (rng() < mix) {
          ph[b] = 2.0 * M_PI * rng();
        } else {
          double sigma = 1.0 / sqrt(beta * refCurv[b]);
          double x = TruncNormal(rng, refMu[b], sigma, lo, hi, z);
          double cosOmega = (cos(x) - cos(th[b]) * cos(th[c])) /
                            (sin(th[b]) * sin(th[c]));
          double omega = acos(std::max(-1.0, std::min(1.0, cosOmega)));
          double rPlus = RestFactor(b, beta, th, ph, ph[c] + omega, true);
          double rMinus = RestFactor(b, beta, th, ph, ph[c] - omega, true);
          double pPlus =
              (rPlus + kSignFloor) / (rPlus + rMinus + 2.0 * kSignFloor);
          ph[b] = ph[c] + (rng() < pPlus ? omega : -omega);
        }
      } else {
        ph[b] = 2.0 * M_PI * rng();
      }

      // exact Boltzmann rejection on the remaining free pair angles
      accepted = rng() < RestFactor(b, beta, th, ph, ph[b], gaussian);
    }
    if (accepted)
      return true;
  }
  return false;
}

inline double BranchSampler::Weight(double beta, const double *th,
                                    const double *ph) {
  double w = 1.0;
  for (uint b = 1; b < nBonds; ++b) {
    if (!UseGaussian(b, beta))
      continue;
    const uint c = ref[b];
    const uint k = kinds[b][c];
    double x = AngleBetween(th[b], ph[b], th[c], ph[c]);
    w *= exp(-beta * (ang->Calc(k, x) - Table(k, beta).uMin)) /
         PhiDensity(b, beta, th, ph);
  }
  return w;
}

inline bool BranchSampler::SolveFixedPhi(uint b, const double *th,
                                         double *ph) {
  for (uint c = 0; c < b; ++c) {
    uint k = kinds[b][c];
    if (!ang->AngleFixed(k))
      continue;
    double cosTerm = cos(th[b]) * cos(th[c]);
    double sinTerm = sin(th[b]) * sin(th[c]);
    double var = (cos(ang->Angle(k)) - cosTerm) / sinTerm;
    // To fix the numerical problem for flat molecule
    var = (var > 1.0 && var < 1.1 ? 1.0 : var);
    var = (var < -1.0 && var > -1.1 ? -1.0 : var);
    ph[b] = acos(var) + ph[c];
    if (std::isnan(ph[b])) {
      failedFixed = b;
      return false;
    }
  }
  return true;
}

} // namespace cbmc

#endif /*BRANCHSAMPLER_H*/
