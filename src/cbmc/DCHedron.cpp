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
  sampler.Init(nBonds, angleKinds, data->ff.angles, &data->bendTables);

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

// Report the atoms around which no branch geometry could be generated
void DCHedron::BranchFailure(TrialMol &mol) const {
  const std::vector<std::string> &names = mol.GetKind().atomTypeNames;
  int b = sampler.FailedFixedPair();
  if (b >= 0) {
    std::cout << "Error: Cannot constrain fix angle for "
              << names[bonded[b]] << " " << names[focus] << " !\n";
  } else {
    std::cout << "Error: CBMC could not generate bond angles around atom "
              << names[focus] << ". The bond angle parameters around this "
              << "atom may be inconsistent.\n";
  }
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
      double bfcTheta = BranchSampler::AngleBetween(th[b], ph[b], th[c], ph[c]);
      bend += ang.Calc(angleKinds[b][c], bfcTheta);
      distSq = mol.AngleDist(bLen[b], bLen[c], bfcTheta);
      nb13 += data->calc.IntraEnergy_1_3(distSq, bonded[b], bonded[c], molIndex);
    }
  }
}

// Draw nAngleTrials branch geometries; select one with probability
// proportional to its weight and accumulate the Rosenbluth weight. Each
// trial's weight is its importance weight from the sampler times the 1-3
// Boltzmann factor. useOld selects the old configuration's bond lengths.
void DCHedron::SampleTrials(TrialMol &mol, uint molIndex, bool useOld) {
  const double beta = data->ff.beta;
  double th[MAX_BONDS], ph[MAX_BONDS], bend, nb13;
  weight = 0.0;
  for (uint i = 0; i < data->nAngleTrials; ++i) {
    if (!sampler.Sample(data->prng, beta, th, ph))
      BranchFailure(mol);
    BranchEnergy(mol, molIndex, th, ph, useOld, bend, nb13);
    double w = sampler.Weight(beta, th, ph) * exp(-beta * nb13);
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

void DCHedron::PrepareNew(TrialMol &newMol, uint molIndex) {
  SampleTrials(newMol, molIndex, false);
}

void DCHedron::PrepareTrialOld(TrialMol &oldMol, uint molIndex) {
  SampleTrials(oldMol, molIndex, true);
}

void DCHedron::SetBranch(const double *th, const double *ph, double bend,
                         double nb13, double w) {
  std::copy(th, th + nBonds, theta);
  std::copy(ph, ph + nBonds, phi);
  bendEnergy = bend;
  oneThree = nb13;
  weight = w;
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
      double bfcTheta =
          BranchSampler::AngleBetween(theta[b], phi[b], theta[c], phi[c]);
      bendEnergy += ff.angles->Calc(angleKinds[b][c], bfcTheta);
      distSq = oldMol.OldDistSq(bonded[c], bonded[b]);
      oneThree +=
          data->calc.IntraEnergy_1_3(distSq, bonded[c], bonded[b], molIndex);
    }
  }
}

// Calculate OldMol bend energy and the old Rosenbluth weight from the actual
// branch plus nTrials freshly generated branches
void DCHedron::ConstrainedAnglesOld(uint nTrials, TrialMol &oldMol,
                                    uint molIndex) {
  IncorporateOld(oldMol, molIndex);
  const double beta = data->ff.beta;
  weight = sampler.Weight(beta, theta, phi) * exp(-beta * oneThree);
  double th[MAX_BONDS], ph[MAX_BONDS], bend, nb13;
  for (uint i = 0; i < nTrials; ++i) {
    if (!sampler.Sample(data->prng, beta, th, ph))
      BranchFailure(oldMol);
    BranchEnergy(oldMol, molIndex, th, ph, true, bend, nb13);
    weight += sampler.Weight(beta, th, ph) * exp(-beta * nb13);
  }
}
} // namespace cbmc
