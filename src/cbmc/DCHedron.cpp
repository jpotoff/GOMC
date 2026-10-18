/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#include "DCHedron.h"

#include <algorithm>
#include <cassert>

#include "DCData.h"
#include "Forcefield.h"
#include "MolSetup.h"
#include "NumLib.h"
#include "PRNG.h"
#include "TrialMol.h"

namespace {
// Wish I could use lambdas..
struct FindA1 {
  FindA1(uint x) : x(x){};
  bool operator()(const mol_setup::Bond &b) { return (b.a1 == x); }
  uint x;
};

struct FindAngle {
  FindAngle(uint x, uint y) : x(x), y(y) {}
  uint x, y;
  bool operator()(const mol_setup::Angle &a) {
    return (a.a0 == x && a.a2 == y) || (a.a0 == y && a.a2 == x);
  }
};

} // namespace

namespace cbmc {

DCHedron::DCHedron(DCData *data, const mol_setup::MolKind &kind, uint focus,
                   uint prev)
    : data(data), focus(focus), prev(prev) {
  using namespace mol_setup;
  std::vector<Bond> onFocus = AtomBonds(kind, focus);
  onFocus.erase(remove_if(onFocus.begin(), onFocus.end(), FindA1(prev)),
                onFocus.end());
  nBonds = onFocus.size();

  for (uint i = 0; i < nBonds; ++i) {
    bonded[i] = onFocus[i].a1;
  }

  std::vector<Angle> angles = AtomMidAngles(kind, focus);
  for (uint i = 0; i < nBonds; ++i) {
    typedef std::vector<Angle>::const_iterator Aiter;
    Aiter free =
        find_if(angles.begin(), angles.end(), FindAngle(prev, bonded[i]));
    assert(free != angles.end());
    angleKinds[i][i] = free->kind;

    for (uint j = i + 1; j < nBonds; ++j) {
      Aiter pair = find_if(angles.begin(), angles.end(),
                           FindAngle(bonded[i], bonded[j]));
      angleKinds[i][j] = pair->kind;
      angleKinds[j][i] = pair->kind;
    }
  }

  phi[0] = 0.0;
  weight = 1.0;
  bendEnergy = oneThree = 0.0;

  if (data->nAngleTrials < 1) {
    std::cout << "Error: CBMC angle trials must be greater than 0.\n";
    exit(EXIT_FAILURE);
  }
}

void DCHedron::SetBondNew(double const *bondLen, double const &anchBond) {
  for (uint i = 0; i < nBonds; ++i) {
    bondLength[i] = bondLen[i];
  }
  anchorBond = anchBond;
}

void DCHedron::SetBondOld(double const *bondLen, double const &anchBond) {
  for (uint i = 0; i < nBonds; ++i) {
    bondLengthOld[i] = bondLen[i];
  }
  anchorBondOld = anchBond;
}

namespace {
const uint kBendBins = 1024;
const uint kBendSub = 8;               // sub-samples per bin for the bound
const double kBendSafety = 1.001;      // margin on the per-bin bound
const unsigned long kMaxBranchTries = 1000000;

// Angle between bonded atoms b and c from their spherical coordinates
// about the focus (polar axis along focus->prev).
inline double AngleBetween(double tb, double pb, double tc, double pc) {
  double x = sin(tb) * sin(tc) * cos(pb - pc) + cos(tb) * cos(tc);
  return acos(std::max(-1.0, std::min(1.0, x)));
}
} // namespace

// Build (or rebuild if beta changed) the sampling table for an angle kind.
const BendTable &DCHedron::GetTable(uint kind) {
  if (data->bendTables.size() <= kind)
    data->bendTables.resize(kind + 1);
  BendTable &t = data->bendTables[kind];
  const double beta = data->ff.beta;
  if (t.beta == beta)
    return t;

  const FFAngles &ang = *data->ff.angles;
  t.width = M_PI / kBendBins;
  // minimum of U on [0, pi], including theta0 where harmonic forms vanish
  t.uMin = ang.Calc(kind, ang.Angle(kind));
  for (uint i = 0; i <= kBendBins * kBendSub; ++i)
    t.uMin = std::min(t.uMin, ang.Calc(kind, i * t.width / kBendSub));

  t.pMax.assign(kBendBins, 0.0);
  t.cumMax.assign(kBendBins, 0.0);
  double sum = 0.0;
  for (uint i = 0; i < kBendBins; ++i) {
    double pm = 0.0;
    for (uint s = 0; s <= kBendSub; ++s) {
      double th = (i + double(s) / kBendSub) * t.width;
      pm = std::max(pm, sin(th) * exp(-beta * (ang.Calc(kind, th) - t.uMin)));
    }
    t.pMax[i] = pm * kBendSafety;
    sum += t.pMax[i];
    t.cumMax[i] = sum;
  }
  t.beta = beta;
  return t;
}

// Exact sample of theta from p(theta) ~ sin(theta) exp(-beta U(theta))
double DCHedron::SampleTheta(uint kind) {
  const FFAngles &ang = *data->ff.angles;
  if (ang.AngleFixed(kind))
    return ang.Angle(kind);
  const BendTable &t = GetTable(kind);
  const double beta = data->ff.beta;
  while (true) {
    double r = data->prng() * t.cumMax.back();
    uint bin = std::upper_bound(t.cumMax.begin(), t.cumMax.end(), r) -
               t.cumMax.begin();
    bin = std::min(bin, kBendBins - 1);
    double th = (bin + data->prng()) * t.width;
    double p = sin(th) * exp(-beta * (ang.Calc(kind, th) - t.uMin));
    if (data->prng() * t.pMax[bin] < p)
      return th;
  }
}

// Exact joint sample of all branch directions (see DCHedron.h). Proposes
// theta_b from its marginal and phi_b uniformly, and accepts with
// exp(-beta sum_{c<b} (U(c,focus,b) - uMin)). The test is applied atom by
// atom and any rejection restarts the whole branch, which keeps it exact.
// Twist angles constrained by a fixed bond angle are solved for as before.
void DCHedron::SampleBranch(TrialMol &mol, double *th, double *ph) {
  const FFAngles &ang = *data->ff.angles;
  const double beta = data->ff.beta;
  for (unsigned long tries = 0; tries < kMaxBranchTries; ++tries) {
    bool accepted = true;
    for (uint b = 0; b < nBonds && accepted; ++b) {
      th[b] = SampleTheta(angleKinds[b][b]);
      if (b == 0) {
        ph[0] = 0.0;
        continue;
      }
      ph[b] = data->prng.rand(2.0 * M_PI);

      // modify the twist angle if it was fixed
      for (uint c = 0; c < b; ++c) {
        if (ang.AngleFixed(angleKinds[b][c])) {
          double cosTerm = cos(th[b]) * cos(th[c]);
          double sinTerm = sin(th[b]) * sin(th[c]);
          double bfcTheta = ang.Angle(angleKinds[b][c]);
          double var = (cos(bfcTheta) - cosTerm) / sinTerm;
          // To fix the numerical problem for flat molecule
          var = (var > 1.0 && var < 1.1 ? 1.0 : var);
          var = (var < -1.0 && var > -1.1 ? -1.0 : var);
          ph[b] = acos(var) + ph[c];
          if (std::isnan(ph[b])) {
            std::cout << "Error: Cannot constrain fix angle for "
                      << mol.GetKind().atomTypeNames[bonded[b]] << " "
                      << mol.GetKind().atomTypeNames[focus] << " "
                      << mol.GetKind().atomTypeNames[bonded[c]] << " !\n";
            exit(EXIT_FAILURE);
          }
        }
      }

      double en = 0.0;
      for (uint c = 0; c < b; ++c) {
        uint k = angleKinds[b][c];
        if (!ang.AngleFixed(k))
          en += ang.Calc(k, AngleBetween(th[b], ph[b], th[c], ph[c])) -
                GetTable(k).uMin;
      }
      accepted = data->prng() < exp(-beta * en);
    }
    if (accepted)
      return;
  }
  std::cout << "Error: CBMC could not generate bond angles around atom "
            << mol.GetKind().atomTypeNames[focus] << " after "
            << kMaxBranchTries << " attempts. The bond angle parameters "
            << "around this atom may be inconsistent.\n";
  exit(EXIT_FAILURE);
}

// Bending energy and 1-3 nonbonded energy of a branch geometry
void DCHedron::BranchEnergy(TrialMol &mol, uint molIndex, const double *th,
                            const double *ph, bool useOld, double &bend,
                            double &nb13) {
  const FFAngles &ang = *data->ff.angles;
  const double *bLen = useOld ? bondLengthOld : bondLength;
  const double aLen = useOld ? anchorBondOld : anchorBond;
  bend = 0.0;
  nb13 = 0.0;
  for (uint b = 0; b < nBonds; ++b) {
    bend += ang.Calc(angleKinds[b][b], th[b]);
    double distSq = mol.AngleDist(aLen, bLen[b], th[b]);
    nb13 += data->calc.IntraEnergy_1_3(distSq, prev, bonded[b], molIndex);
    for (uint c = 0; c < b; ++c) {
      double bfcTheta = AngleBetween(th[b], ph[b], th[c], ph[c]);
      bend += ang.Calc(angleKinds[b][c], bfcTheta);
      distSq = mol.AngleDist(bLen[b], bLen[c], bfcTheta);
      nb13 += data->calc.IntraEnergy_1_3(distSq, bonded[b], bonded[c], molIndex);
    }
  }
}

void DCHedron::PrepareNew(TrialMol &newMol, uint molIndex) {
  SampleBranch(newMol, theta, phi);
  BranchEnergy(newMol, molIndex, theta, phi, false, bendEnergy, oneThree);
  weight = 1.0;
  if (!data->ff.OneThree)
    return;

  // Rosenbluth weight the 1-3 interactions over nAngleTrials exact branches
  const double beta = data->ff.beta;
  weight = exp(-beta * oneThree);
  double th[MAX_BONDS], ph[MAX_BONDS], bend, nb13;
  for (uint i = 1; i < data->nAngleTrials; ++i) {
    SampleBranch(newMol, th, ph);
    BranchEnergy(newMol, molIndex, th, ph, false, bend, nb13);
    double w = exp(-beta * nb13);
    weight += w;
    // keep each trial with probability proportional to its weight
    if (data->prng() * weight < w) {
      std::copy(th, th + nBonds, theta);
      std::copy(ph, ph + nBonds, phi);
      bendEnergy = bend;
      oneThree = nb13;
    }
  }
}

void DCHedron::PrepareOld(TrialMol &oldMol, uint molIndex) {
  oneThree = 0.0;
  bendEnergy = 0.0;
  weight = 1.0;
}

// Actual angles and energies of the old branch
void DCHedron::IncorporateOld(TrialMol &oldMol, uint molIndex) {
  bendEnergy = 0.0;
  oneThree = 0.0;
  const Forcefield &ff = data->ff;
  for (uint b = 0; b < nBonds; ++b) {
    oldMol.OldThetaAndPhi(bonded[b], focus, theta[b], phi[b]);
    bendEnergy += ff.angles->Calc(angleKinds[b][b], theta[b]);
    double distSq = oldMol.OldDistSq(prev, bonded[b]);
    oneThree += data->calc.IntraEnergy_1_3(distSq, prev, bonded[b], molIndex);
    for (uint c = 0; c < b; ++c) {
      double bfcTheta = AngleBetween(theta[b], phi[b], theta[c], phi[c]);
      bendEnergy += ff.angles->Calc(angleKinds[b][c], bfcTheta);
      distSq = oldMol.OldDistSq(bonded[c], bonded[b]);
      oneThree +=
          data->calc.IntraEnergy_1_3(distSq, bonded[c], bonded[b], molIndex);
    }
  }
}

// Calculate OldMol bend energy and, if 1-3 interactions are enabled, the
// old Rosenbluth weight from the actual branch plus nTrials exact branches
void DCHedron::ConstrainedAnglesOld(uint nTrials, TrialMol &oldMol,
                                    uint molIndex) {
  IncorporateOld(oldMol, molIndex);
  weight = 1.0;
  if (!data->ff.OneThree)
    return;

  const double beta = data->ff.beta;
  weight = exp(-beta * oneThree);
  double th[MAX_BONDS], ph[MAX_BONDS], bend, nb13;
  for (uint i = 0; i < nTrials; ++i) {
    SampleBranch(oldMol, th, ph);
    BranchEnergy(oldMol, molIndex, th, ph, true, bend, nb13);
    weight += exp(-beta * nb13);
  }
}
} // namespace cbmc
