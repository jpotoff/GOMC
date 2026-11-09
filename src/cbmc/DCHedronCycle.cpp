/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#include "DCHedronCycle.h"

#include <algorithm>
#include <cassert>
#include <numeric>

#include "DCData.h"
#include "Forcefield.h"
#include "Geometry.h"
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

// Check to see if angle a is in the ring or not
bool IsInRing(const std::vector<int> &cycAtoms, const mol_setup::Angle &a) {
  bool res = true;
  if (std::find(cycAtoms.begin(), cycAtoms.end(), a.a0) == cycAtoms.end()) {
    res = false;
  } else if (std::find(cycAtoms.begin(), cycAtoms.end(), a.a1) ==
             cycAtoms.end()) {
    res = false;
  } else if (std::find(cycAtoms.begin(), cycAtoms.end(), a.a2) ==
             cycAtoms.end()) {
    res = false;
  }
  return res;
}

} // namespace

namespace cbmc {

DCHedronCycle::DCHedronCycle(DCData *data, const mol_setup::MolKind &kind,
                             const std::vector<int> &cycAtoms, uint focus,
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
  double sumAngle = 0.0;
  for (uint a = 0; a < angles.size(); a++) {
    sumAngle += data->ff.angles->Angle(angles[a].kind);
  }
  // If sum of angles (sumAngle +/- 10) ~ 2*pi = 6.283, it means they are in a
  // plane To avoid geometric conflict for flexible angle, we consider it fixed
  // and let crankshaft sample it. 0.1745 ~= 10 degree tolerance
  bool angleInPlane = (std::abs(2.0 * M_PI - sumAngle) < 0.1745);
  bool constrainAngInRing = false;

  for (uint i = 0; i < nBonds; ++i) {
    typedef std::vector<Angle>::const_iterator Aiter;
    Aiter free =
        find_if(angles.begin(), angles.end(), FindAngle(prev, bonded[i]));
    assert(free != angles.end());
    angleKinds[i][i] = free->kind;
    angleInRing[i][i] = IsInRing(cycAtoms, *free);

    for (uint j = i + 1; j < nBonds; ++j) {
      Aiter pair = find_if(angles.begin(), angles.end(),
                           FindAngle(bonded[i], bonded[j]));
      angleKinds[i][j] = pair->kind;
      angleKinds[j][i] = pair->kind;
      angleInRing[i][j] = IsInRing(cycAtoms, *pair);
      angleInRing[j][i] = angleInRing[i][j];
      constrainAngInRing |= angleInRing[i][j];
    }
  }

  // If one of the constrained angles belongs to a ring and the angle forms a
  // plane
  // we fix the free and constrained angles
  if (constrainAngInRing && angleInPlane) {
    for (uint i = 0; i < nBonds; ++i) {
      angleInRing[i][i] = true;
      for (uint j = i + 1; j < nBonds; ++j) {
        angleInRing[i][j] = true;
        angleInRing[j][i] = true;
      }
    }
  }

  phi[0] = 0.0;
  weight = 1.0;
  bendEnergy = oneThree = 0.0;
  // ring angles (and angles fixed in the force field) are constrained to the
  // existing geometry; everything else is sampled
  bool constrained[MAX_BONDS][MAX_BONDS];
  for (uint i = 0; i < nBonds; ++i)
    for (uint j = 0; j < nBonds; ++j)
      constrained[i][j] = angleInRing[i][j];
  sampler.Init(nBonds, angleKinds, data->ff.angles, &data->bendTables,
               constrained);

  if (data->nAngleTrials < 1) {
    std::cout << "Error: CBMC angle trials must be greater than 0.\n";
    exit(EXIT_FAILURE);
  }
}

void DCHedronCycle::SetBondNew(double const *bondLen, double const &anchBond) {
  for (uint i = 0; i < nBonds; ++i) {
    bondLength[i] = bondLen[i];
  }
  anchorBond = anchBond;
}

void DCHedronCycle::SetBondOld(double const *bondLen, double const &anchBond) {
  for (uint i = 0; i < nBonds; ++i) {
    bondLengthOld[i] = bondLen[i];
  }
  anchorBondOld = anchBond;
}

double DCHedronCycle::CalcTheta(TrialMol &mol, const uint a0, const uint a1,
                                const uint a2) {
  double theta = 0.0;
  if (mol.AtomExists(a0) && mol.AtomExists(a2)) {
    // Calculate theta using tCoords
    theta = mol.GetTheta(a0, a1, a2);
  } else {
    // Calculate theta using bCoords
    const XYZArray &coords = mol.GetBCoords();
    // Since data in bCoords in unwrap, no need to calc minImage
    theta = geom::Theta(coords.Difference(a0, a1), coords.Difference(a2, a1));
  }
  return theta;
}

// Report the atoms around which no branch geometry could be generated
void DCHedronCycle::BranchFailure(TrialMol &mol) const {
  const std::vector<std::string> &names = mol.GetKind().atomTypeNames;
  int b = sampler.FailedFixedPair();
  if (b >= 0) {
    std::cout << "Error: Cannot construct ring or fixed angle for "
              << names[bonded[b]] << " " << names[focus] << " !\n";
  } else {
    std::cout << "Error: CBMC could not generate bond angles around ring atom "
              << names[focus] << ". The bond angle parameters around this "
              << "atom may be inconsistent.\n";
  }
  exit(EXIT_FAILURE);
}

void DCHedronCycle::SetConstraints(TrialMol &mol) {
  SetBasis(mol, focus, prev);
  for (uint b = 0; b < nBonds; ++b) {
    if (sampler.Constrained(b, b))
      sampler.SetConstraint(b, b, CalcTheta(mol, bonded[b], focus, prev));
    for (uint c = 0; c < b; ++c) {
      if (!sampler.Constrained(b, c))
        continue;
      // keep the handedness of the existing geometry: the sign of the twist
      // of b relative to c about the focus->prev axis
      double d = CalcOldPhi(mol, bonded[b], focus) -
                 CalcOldPhi(mol, bonded[c], focus);
      sampler.SetConstraint(b, c, CalcTheta(mol, bonded[b], focus, bonded[c]),
                            sin(d) < 0.0 ? -1.0 : 1.0);
    }
  }
}

// Bending energy and 1-3 nonbonded energy of a branch geometry
void DCHedronCycle::BranchEnergy(TrialMol &mol, uint molIndex,
                                 const double *th, const double *ph,
                                 bool useOld, double &bend, double &nb13) {
  const FFAngles &ang = *data->ff.angles;
  const double *bLen = useOld ? bondLengthOld : bondLength;
  const double aLen = useOld ? anchorBondOld : anchorBond;
  bend = 0.0;
  nb13 = 0.0;
  for (uint b = 0; b < nBonds; ++b) {
    bend += ang.Calc(angleKinds[b][b], th[b]);
    double distSq = mol.AngleDist(aLen, bLen[b], th[b]);
    if (!mol.CountedPair(prev, bonded[b]))
      nb13 += data->calc.IntraEnergy_1_3(distSq, prev, bonded[b], molIndex);
    for (uint c = 0; c < b; ++c) {
      double bfcTheta = BranchSampler::AngleBetween(th[b], ph[b], th[c], ph[c]);
      bend += ang.Calc(angleKinds[b][c], bfcTheta);
      distSq = mol.AngleDist(bLen[b], bLen[c], bfcTheta);
      if (!mol.CountedPair(bonded[b], bonded[c]))
        nb13 +=
            data->calc.IntraEnergy_1_3(distSq, bonded[b], bonded[c], molIndex);
    }
  }
}

// Draw nTrials branch geometries; select one with probability proportional
// to its weight and accumulate the Rosenbluth weight. Each trial's weight is
// its importance weight from the sampler times the 1-3 Boltzmann factor.
void DCHedronCycle::SampleTrials(TrialMol &mol, uint molIndex, bool useOld,
                                 uint nTrials) {
  const double beta = data->ff.beta;
  double th[MAX_BONDS], ph[MAX_BONDS], bend, nb13;
  weight = 0.0;
  for (uint i = 0; i < nTrials; ++i) {
    double ws;
    if (!sampler.Sample(data->prng, beta, th, ph, &ws))
      BranchFailure(mol);
    BranchEnergy(mol, molIndex, th, ph, useOld, bend, nb13);
    double w = ws * exp(-beta * nb13);
    weight += w;
    // keep each trial with probability proportional to its weight
    if (i == 0 || data->prng() * weight < w) {
      std::copy(th, th + nBonds, theta);
      std::copy(ph, ph + nBonds, phi);
      bendEnergy = bend;
      oneThree = nb13;
    }
  }
}

void DCHedronCycle::PrepareNew(TrialMol &newMol, uint molIndex) {
  SetConstraints(newMol);
  SampleTrials(newMol, molIndex, false, data->nAngleTrials);
}

void DCHedronCycle::PrepareOld(TrialMol &oldMol, uint molIndex) {
  oneThree = 0.0;
  bendEnergy = 0.0;
  weight = 1.0;
}

// Actual angles and energies of the old branch
void DCHedronCycle::IncorporateOld(TrialMol &oldMol, uint molIndex) {
  bendEnergy = 0.0;
  oneThree = 0.0;
  const Forcefield &ff = data->ff;
  for (uint b = 0; b < nBonds; ++b) {
    oldMol.OldThetaAndPhi(bonded[b], focus, theta[b], phi[b]);
    bendEnergy += ff.angles->Calc(angleKinds[b][b], theta[b]);
    double distSq = oldMol.OldDistSq(prev, bonded[b]);
    if (!oldMol.CountedPair(prev, bonded[b]))
      oneThree +=
          data->calc.IntraEnergy_1_3(distSq, prev, bonded[b], molIndex);
    for (uint c = 0; c < b; ++c) {
      double bfcTheta =
          BranchSampler::AngleBetween(theta[b], phi[b], theta[c], phi[c]);
      bendEnergy += ff.angles->Calc(angleKinds[b][c], bfcTheta);
      distSq = oldMol.OldDistSq(bonded[c], bonded[b]);
      if (!oldMol.CountedPair(bonded[c], bonded[b]))
        oneThree +=
            data->calc.IntraEnergy_1_3(distSq, bonded[c], bonded[b], molIndex);
    }
  }
}

// Calculate OldMol bend energy and the old Rosenbluth weight from the actual
// branch plus nTrials freshly generated branches. The caller has set the
// oldMol basis (focus, prev).
void DCHedronCycle::ConstrainedAnglesOld(uint nTrials, TrialMol &oldMol,
                                         uint molIndex) {
  SetConstraints(oldMol);
  IncorporateOld(oldMol, molIndex);
  const double beta = data->ff.beta;
  weight = sampler.Weight(beta, theta, phi) * exp(-beta * oneThree);
  double th[MAX_BONDS], ph[MAX_BONDS], bend, nb13;
  for (uint i = 0; i < nTrials; ++i) {
    double ws;
    if (!sampler.Sample(data->prng, beta, th, ph, &ws))
      BranchFailure(oldMol);
    BranchEnergy(oldMol, molIndex, th, ph, true, bend, nb13);
    weight += ws * exp(-beta * nb13);
  }
}

// Record the 1-3 pairs this step added, so that a pair reached by a second
// angle (C1-C3 in a four-membered ring) is not counted again
void DCHedronCycle::MarkCounted13(TrialMol &mol) const {
  if (!mol.OneThree())
    return;
  for (uint b = 0; b < nBonds; ++b) {
    mol.MarkCountedPair(prev, bonded[b]);
    for (uint c = 0; c < b; ++c)
      mol.MarkCountedPair(bonded[b], bonded[c]);
  }
}

void DCHedronCycle::SetBasis(TrialMol &mol, uint p1, uint p2) {
  // Calculate the dihedral using bCoords
  const XYZArray &coords = mol.GetBCoords();
  // W is unit vec of p1->p2
  XYZ wVec = coords.Difference(p2, p1);
  wVec.Normalize();
  XYZ uVec;
  // check to make sure our W isn't in line with the standard X Axis
  if (std::abs(wVec.x) < 0.8) {
    // V will be W x the standard X unit vec
    uVec = XYZ(1.0, 0.0, 0.0);
  } else {
    // V will be W x the standard Y unit vec
    uVec = XYZ(0.0, 1.0, 0.0);
  }
  XYZ vVec = geom::Cross(wVec, uVec);
  vVec.Normalize();
  // U is unit vec perpendicular to both V and W
  uVec = geom::Cross(vVec, wVec);
  growthToWorld.BasisRotation(uVec, vVec, wVec);
  worldToGrowth = growthToWorld.Inverse();
}

double DCHedronCycle::CalcOldPhi(TrialMol &mol, uint atom,
                                 uint lastAtom) const {
  // Calculate the dihedral using bCoords
  const XYZArray &coords = mol.GetBCoords();
  XYZ diff = coords.Difference(atom, lastAtom);
  XYZ growthCoords = worldToGrowth.Apply(diff);
  return atan2(growthCoords.y, growthCoords.x);
}

} // namespace cbmc
