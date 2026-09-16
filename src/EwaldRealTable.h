/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) 2.80
Copyright (C) 2022 GOMC Group
A copy of the MIT License can be found in License.txt
along with this program, also can be found at
<https://opensource.org/licenses/MIT>.
********************************************************************************/
#ifndef EWALD_REAL_TABLE_H
#define EWALD_REAL_TABLE_H

#include "BasicTypes.h"
#include "EnsemblePreprocessor.h"
#include <cmath>
#include <cstdio>
#include <vector>

//
// Tabulated real-space Ewald kernels.
//
// Every forcefield flavor evaluates the same two quantities in its `ewald`
// branch, and they are functions of distSq alone once the box is fixed:
//
//   energy:  erfc(alpha*r) / r
//   virial: (erfc(alpha*r) / r + alpha*(2/sqrt(pi))*exp(-alpha^2*r^2)) / r^2
//
// Tabulating them on distSq removes the erfc, the exp, the sqrt and the
// division in one lookup. Each interval stores four cubic coefficients
// contiguously, so an evaluation is a single 32-byte read and three FMAs.
//
// Accuracy note: alpha = sqrt(-log(tolerance))/rCutCoulomb, so alpha*rCut is
// fixed by the Ewald tolerance alone and the tabulated domain has the same
// shape for every box and cutoff. The grid is therefore specified by interval
// *width* rather than interval count -- a fixed count would make accuracy
// depend on the user's cutoff, since a larger rCut widens the distSq domain and
// coarsens the spacing (cubic error grows as dx^4). At DEFAULT_DX the maximum
// relative error is ~2e-7 independent of cutoff, two orders of magnitude below
// the 1e-5 real-space truncation error the tolerance already accepts.
//
class EwaldRealTable {
public:
  // Interval width in distSq (Angstrom^2). 0.05 gives ~62 KiB per box per
  // kernel at a 10 A cutoff and ~122 KiB at 14 A -- small enough to stay
  // cache-resident, with ~2e-7 max relative error at any cutoff.
  static double DefaultDx() { return 0.05; }

  // The table grows as rCutCoulomb^2 (~0.2 MB at 12 A, ~0.5 MB at 20 A,
  // ~12 MB at 100 A). Past this cutoff it no longer stays cache-resident and
  // the lookups cost more than the erfc they replace, so those boxes keep the
  // stock erfc. Typical GEMC gas-phase cutoffs land above this.
  static double MaxCutoff() { return 25.0; }

  EwaldRealTable() : built(false), dx(0.0) {}

  // Safe to call repeatedly; rebuilds if alpha or the cutoff changed.
  void Init(const double *alpha, const double *rCutCoulombSq, double rCutLowSq,
            double intervalWidth = 0.05) {
    dx = intervalWidth;
    const double maxCutSq = MaxCutoff() * MaxCutoff();
    // Below r2Lo the kernel diverges and callers fall back to the exact form.
    // Anything that close is an overlap and gets rejected, so the floor only
    // has to keep the table away from the singularity.
    for (uint b = 0; b < BOX_TOTAL; ++b) {
      if (rCutCoulombSq[b] > maxCutSq) {
        // Leaving the grid empty makes every lookup miss, so the callers take
        // their exact-erfc path for this box.
        box[b].Clear();
        printf("Info: Box %d  Tabulated Ewald real space Inactive: "
               "RcutCoulomb %.1f A exceeds %.1f A, using standard erfc\n",
               (int)b, std::sqrt(rCutCoulombSq[b]), MaxCutoff());
        continue;
      }
      box[b].Build(alpha[b], rCutCoulombSq[b], std::max(rCutLowSq, 0.25),
                   intervalWidth);
    }
    built = true;
  }

  bool IsBuilt() const { return built; }

  // erfc(alpha*r)/r. Returns false if distSq is outside the table, in which
  // case the caller must use the exact expression.
  inline bool Energy(const double distSq, const uint b, double &out) const {
    return box[b].Eval(box[b].en, distSq, out);
  }

  // (erfc(alpha*r)/r + alpha*2/sqrt(pi)*exp(-alpha^2 r^2)) / r^2
  inline bool Virial(const double distSq, const uint b, double &out) const {
    return box[b].Eval(box[b].vir, distSq, out);
  }

  // Exact forms, used below the table floor and by the unit tests.
  static double ExactEnergy(const double distSq, const double alpha) {
    const double r = std::sqrt(distSq);
    return std::erfc(alpha * r) / r;
  }
  static double ExactVirial(const double distSq, const double alpha) {
    const double r = std::sqrt(distSq);
    const double alphaSq = alpha * alpha;
    return (std::erfc(alpha * r) / r +
            alpha * M_2_SQRTPI * std::exp(-alphaSq * distSq)) /
           distSq;
  }

private:
  struct Grid {
    std::vector<double> en;  // 4 cubic coefficients per interval
    std::vector<double> vir;
    double lo, invDx;
    int n;

    Grid() : lo(0.0), invDx(0.0), n(0) {}

    void Clear() {
      n = 0;
      en.clear();
      vir.clear();
    }

    void Build(double alpha, double r2Hi, double r2Lo, double intervalWidth) {
      lo = r2Lo;
      // A box with electrostatics off has rCutCoulombSq == 0; leave it empty
      // so every lookup reports a miss and takes the exact path.
      if (r2Hi <= r2Lo || intervalWidth <= 0.0) {
        Clear();
        return;
      }
      // Fixed spacing, so accuracy does not depend on the cutoff.
      n = (int)std::ceil((r2Hi - r2Lo) / intervalWidth);
      if (n < 64)
        n = 64;
      const double dx = (r2Hi - r2Lo) / n;
      invDx = 1.0 / dx;
      en.resize(4 * n);
      vir.resize(4 * n);
      for (int i = 0; i < n; ++i) {
        const double x0 = lo + i * dx;
        Hermite(&en[4 * i], ExactEnergy(x0, alpha), ExactEnergy(x0 + dx, alpha),
                dEnergy(x0, alpha) * dx, dEnergy(x0 + dx, alpha) * dx);
        Hermite(&vir[4 * i], ExactVirial(x0, alpha), ExactVirial(x0 + dx, alpha),
                dVirial(x0, alpha) * dx, dVirial(x0 + dx, alpha) * dx);
      }
    }

    inline bool Eval(const std::vector<double> &c, const double distSq,
                     double &out) const {
      const double t = (distSq - lo) * invDx;
      // Catches distSq below the floor, above the cutoff, and n == 0.
      if (t < 0.0 || t >= (double)n)
        return false;
      const int i = (int)t;
      const double f = t - i;
      const double *p = &c[4 * i];
      out = p[0] + f * (p[1] + f * (p[2] + f * p[3]));
      return true;
    }

    // Cubic Hermite on [0,1] expanded into plain polynomial coefficients.
    static void Hermite(double *c, double v0, double v1, double m0, double m1) {
      c[0] = v0;
      c[1] = m0;
      c[2] = -3.0 * v0 + 3.0 * v1 - 2.0 * m0 - m1;
      c[3] = 2.0 * v0 - 2.0 * v1 + m0 + m1;
    }

    // d/d(distSq) of the tabulated functions.
    static double dEnergy(double r2, double alpha) {
      const double r = std::sqrt(r2);
      const double dErfc =
          -2.0 * alpha * std::exp(-alpha * alpha * r2) / std::sqrt(M_PI);
      return (dErfc / r - std::erfc(alpha * r) / r2) * (0.5 / r);
    }
    static double dVirial(double r2, double alpha) {
      const double r = std::sqrt(r2);
      const double aSq = alpha * alpha;
      const double e = std::exp(-aSq * r2);
      const double erfcTerm = std::erfc(alpha * r) / r;
      const double g = erfcTerm + alpha * M_2_SQRTPI * e;
      // dg/d(r2)
      const double dErfc = -2.0 * alpha * e / std::sqrt(M_PI);
      const double dErfcTerm = (dErfc / r - erfcTerm / r) * (0.5 / r);
      const double dg = dErfcTerm + alpha * M_2_SQRTPI * (-aSq) * e;
      return (dg * r2 - g) / (r2 * r2);
    }
  };

  bool built;
  double dx;
  Grid box[BOX_TOTAL];
};

#endif /*EWALD_REAL_TABLE_H*/
