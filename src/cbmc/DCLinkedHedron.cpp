/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#include "DCLinkedHedron.h"

#include <cassert>
#include <numeric>

#include "DCData.h"
#include "Forcefield.h"
#include "MolSetup.h"
#include "NumLib.h"
#include "PRNG.h"
#include "TrialMol.h"

namespace {
struct FindA1 {
  FindA1(uint x) : x(x){};
  bool operator()(const mol_setup::Bond &b) { return (b.a1 == x); }
  uint x;
};

struct FindDih {
  FindDih(uint x, uint y) : x(x), y(y) {}
  uint x, y;
  bool operator()(const mol_setup::Dihedral d) {
    return (d.a0 == x && d.a3 == y) || (d.a0 == y && d.a3 == x);
  }
};

} // namespace

namespace cbmc {
DCLinkedHedron::DCLinkedHedron(DCData *data, const mol_setup::MolKind &kind,
                               uint focus, uint prev)
    : data(data), hed(data, kind, focus, prev) {
  using namespace mol_setup;
  std::vector<Bond> onFocus = AtomBonds(kind, hed.Focus());
  onFocus.erase(remove_if(onFocus.begin(), onFocus.end(), FindA1(prev)),
                onFocus.end());
  // Find the atoms bonded to focus, except prev
  for (uint i = 0; i < hed.NumBond(); ++i) {
    bondKinds[i] = onFocus[i].kind;
  }

  std::vector<Bond> onPrev = AtomBonds(kind, hed.Prev());
  onPrev.erase(remove_if(onPrev.begin(), onPrev.end(), FindA1(hed.Focus())),
               onPrev.end());
  nPrevBonds = onPrev.size();

  for (uint i = 0; i < nPrevBonds; ++i) {
    prevBonded[i] = onPrev[i].a1;
  }

  std::vector<Dihedral> dihs = DihsOnBond(kind, hed.Focus(), hed.Prev());
  for (uint i = 0; i < hed.NumBond(); ++i) {
    for (uint j = 0; j < nPrevBonds; ++j) {
      std::vector<Dihedral>::const_iterator match = find_if(
          dihs.begin(), dihs.end(), FindDih(hed.Bonded(i), prevBonded[j]));
      assert(match != dihs.end());
      dihKinds[i][j] = match->kind;
    }
  }

  if (data->nLJTrialsNth < 1) {
    std::cout << "Error: CBMC secondary atom trials must be greater than 0.\n";
    exit(EXIT_FAILURE);
  }

  if (data->nDihTrials < 1) {
    std::cout << "Error: CBMC dihedral trials must be greater than 0.\n";
    exit(EXIT_FAILURE);
  }
}

void DCLinkedHedron::PrepareNew(TrialMol &newMol, uint molIndex) {
  // Get new bond information
  SetBondLengthNew(newMol);
  hed.SetBondNew(bondLength, anchorBond);
  bondEnergy = 0.0;
  for (uint i = 0; i < hed.NumBond(); ++i) {
    bondEnergy += data->ff.bonds.Calc(bondKinds[i], bondLength[i]);
  }
}

void DCLinkedHedron::PrepareOld(TrialMol &oldMol, uint molIndex) {
  // Get old bond information
  SetBondLengthOld(oldMol);
  hed.SetBondOld(bondLengthOld, anchorBondOld);
  hed.PrepareOld(oldMol, molIndex);
  bondEnergy = 0.0;
  for (uint i = 0; i < hed.NumBond(); ++i) {
    bondEnergy += data->ff.bonds.Calc(bondKinds[i], bondLengthOld[i]);
  }
}

void DCLinkedHedron::SetBondLengthNew(TrialMol &newMol) {
  for (uint i = 0; i < hed.NumBond(); ++i) {
    bondLength[i] = data->ff.bonds.Length(bondKinds[i]);
  }
  // anchorBond is built, we need the actual length
  anchorBond = sqrt(newMol.OldDistSq(hed.Focus(), hed.Prev()));
}

void DCLinkedHedron::SetBondLengthOld(TrialMol &oldMol) {
  for (uint i = 0; i < hed.NumBond(); ++i) {
    bondLengthOld[i] = sqrt(oldMol.OldDistSq(hed.Focus(), hed.Bonded(i)));
  }
  anchorBondOld = sqrt(oldMol.OldDistSq(hed.Focus(), hed.Prev()));
}

void DCLinkedHedron::BuildNew(TrialMol &newMol, uint molIndex) {
  PRNG &prng = data->prng;
  // const CalculateEnergy& calc = data->calc;
  // const Forcefield& ff = data->ff;
  uint nLJTrials = data->nLJTrialsNth;
  uint nDihTrials = data->nDihTrials;
  double *torsion = data->angles;
  double *torWeights = data->angleWeights;
  double *torEnergy = data->angleEnergy;
  double *ljWeights = data->ljWeights;
  double *bondedEn = data->bonded;
  double *inter = data->inter;
  double *nonbonded = data->nonbonded;
  double *nonbonded_1_4 = data->nonbonded_1_4;
  double *real = data->real;
  double *oneFour = data->oneFour;
  bool *overlap = data->overlap;

  std::fill_n(ljWeights, nLJTrials, 0.0);
  std::fill_n(bondedEn, nLJTrials, 0.0);
  std::fill_n(oneFour, nLJTrials, 0.0);
  std::fill_n(overlap, nLJTrials, false);

  // get info about existing geometry
  double prevPhi[MAX_BONDS];
  RotationMatrix cross, tensor;
  SetupBasis(newMol, prevPhi, cross, tensor);
  const XYZ center = newMol.AtomPosition(hed.Focus());
  XYZArray *positions = data->multiPositions;

  // counting backward to preserve prototype
  for (uint lj = nLJTrials; lj-- > 0;) {
    // Each LJ trial gets its own branch geometry, so the bend angles and
    // hydrogen positions are selected together with the torsion and the
    // nonbonded energy. The branch's Rosenbluth weight is part of the trial
    // weight.
    hed.PrepareNew(newMol, molIndex);
    SaveBranch(lj);
    for (uint b = 0; b < hed.NumBond(); ++b) {
      // get position and shift to origin
      positions[b].Set(
          0, newMol.RawRectCoords(bondLength[b], hed.Theta(b), hed.Phi(b)));
    }
    ChooseTorsion(newMol, molIndex, prevPhi, cross, tensor);
    double torSum = std::accumulate(torWeights, torWeights + nDihTrials, 0.0);
    ljWeights[lj] = hed.GetWeight() * torSum;
    uint winner = prng.PickWeighted(torWeights, nDihTrials, torSum);
    bondedEn[lj] = torEnergy[winner];
    oneFour[lj] = nonbonded_1_4[winner];
    // convert chosen torsion to 3D positions
    RotationMatrix spin =
        RotationMatrix::FromAxisAngle(-torsion[winner], cross, tensor);
    for (uint b = 0; b < hed.NumBond(); ++b) {
      // find positions
      positions[b].Set(lj, spin.Apply(positions[b][0]));
      positions[b].Add(lj, center);
    }
  }

  for (uint b = 0; b < hed.NumBond(); ++b) {
    data->axes.WrapPBC(positions[b], newMol.GetBox());
  }

  GOMC_INSTR(double instrB =
                 std::accumulate(ljWeights, ljWeights + nLJTrials, 0.0);)
  double stepWeight = EvalLJ(newMol, molIndex);
  double factor = stepWeight / nLJTrials;
  uint winner;
  if (data->lookAhead > 0 && !children.empty()) {
    // one-step look-ahead: select with w_j v_j and weigh the step
    // (sum_j w_j v_j) / (k v_c); exact for any v (LookAheadCBMCTest)
    laV.resize(nLJTrials);
    laSel.resize(nLJTrials);
    LookAhead(newMol, molIndex, nLJTrials, laV.data());
    double sel = 0.0;
    for (uint j = 0; j < nLJTrials; ++j) {
      laSel[j] = ljWeights[j] * laV[j];
      sel += laSel[j];
    }
    winner = prng.PickWeighted(laSel.data(), nLJTrials, sel);
    factor = sel > 0.0 ? sel / (nLJTrials * laV[winner]) : 0.0;
    GOMC_INSTR(instr::RecordStageFactor(newMol, instr::LATER, laSel.data(),
                                        nLJTrials, overlap, factor);)
  } else {
    GOMC_INSTR(instr::RecordStage(newMol, instr::LATER, ljWeights, nLJTrials, overlap));
    winner = prng.PickWeighted(ljWeights, nLJTrials, stepWeight);
  }
  GOMC_INSTR(instr::RecordBondedPart(newMol, std::log(instrB / nLJTrials));)
  RestoreBranch(winner);
#ifdef GOMC_INSTRUMENT
  if (instr::Collector::Get().On(instr::WEIGHTS)) {
    std::vector<instr::PairE> pairs;
    for (uint b = 0; b < hed.NumBond(); ++b) {
      data->calc.ParticleNonbondedPairs(newMol, positions[b][winner],
                                        hed.Bonded(b), newMol.GetBox(), pairs);
      instr::RecordPairs(newMol, hed.Bonded(b), pairs);
    }
  }
#endif
  for (uint b = 0; b < hed.NumBond(); ++b) {
    newMol.AddAtom(hed.Bonded(b), positions[b][winner]);
    newMol.AddBonds(hed.Bonded(b), hed.Focus());
  }
  newMol.UpdateOverlap(overlap[winner]);
  newMol.AddEnergy(
      Energy(bondedEn[winner] + hed.GetEnergy() + bondEnergy,
             nonbonded[winner] + hed.GetNonBondedEn() + oneFour[winner],
             inter[winner], real[winner], 0.0, 0.0, 0.0));
  newMol.MultWeight(factor);
}

void DCLinkedHedron::BuildOld(TrialMol &oldMol, uint molIndex) {
  PRNG &prng = data->prng;
  // const CalculateEnergy& calc = data->calc;
  uint nLJTrials = data->nLJTrialsNth;
  uint nDihTrials = data->nDihTrials;
  double *torsion = data->angles;
  double *torWeights = data->angleWeights;
  double *torEnergy = data->angleEnergy;
  double *ljWeights = data->ljWeights;
  double *bondedEn = data->bonded;
  double *inter = data->inter;
  double *nonbonded = data->nonbonded;
  double *nonbonded_1_4 = data->nonbonded_1_4;
  double *real = data->real;
  double *oneFour = data->oneFour;
  bool *overlap = data->overlap;

  std::fill_n(ljWeights, nLJTrials, 0.0);
  std::fill_n(bondedEn, nLJTrials, 0.0);
  std::fill_n(oneFour, nLJTrials, 0.0);
  std::fill_n(oneFour, nLJTrials, 0.0);
  std::fill_n(overlap, nLJTrials, false);

  // get info about existing geometry
  oldMol.SetBasis(hed.Focus(), hed.Prev());
  // Actual branch angles and energies, and the branch Rosenbluth weight of
  // trial 0 (actual branch plus nAngleTrials - 1 generated ones)
  hed.ConstrainedAnglesOld(data->nAngleTrials - 1, oldMol, molIndex);
  const XYZ center = oldMol.AtomPosition(hed.Focus());
  XYZArray *positions = data->multiPositions;
  double prevPhi[MAX_BONDS];
  for (uint i = 0; i < hed.NumBond(); ++i) {
    // get position and shift to origin
    positions[i].Set(0, oldMol.AtomPosition(hed.Bonded(i)));
    data->axes.UnwrapPBC(positions[i], 0, 1, oldMol.GetBox(), center);
    positions[i].Add(0, -center);
  }
  for (uint i = 0; i < nPrevBonds; ++i) {
    double t;
    // not using theta, so this is a wasted cos and sqrt
    oldMol.OldThetaAndPhi(prevBonded[i], hed.Prev(), t, prevPhi[i]);
  }
  XYZ rotationAxis =
      oldMol.AtomPosition(hed.Focus()) - oldMol.AtomPosition(hed.Prev());
  rotationAxis = data->axes.MinImage(rotationAxis, oldMol.GetBox());
  rotationAxis *= (1 / rotationAxis.Length());
  RotationMatrix cross = RotationMatrix::CrossProduct(rotationAxis);
  RotationMatrix tensor = RotationMatrix::TensorProduct(rotationAxis);

  // keep the actual branch (trial 0) while generating the other trials
  SaveBranch(0);
  XYZ actualBase[MAX_BONDS];
  for (uint b = 0; b < hed.NumBond(); ++b)
    actualBase[b] = positions[b][0];

  // counting backward to preserve prototype
  for (uint lj = nLJTrials; lj-- > 1;) {
    // fresh branch geometry for each trial, as in BuildNew
    hed.PrepareTrialOld(oldMol, molIndex);
    for (uint b = 0; b < hed.NumBond(); ++b) {
      positions[b].Set(0, oldMol.RawRectCoords(bondLengthOld[b], hed.Theta(b),
                                               hed.Phi(b)));
    }
    ChooseTorsion(oldMol, molIndex, prevPhi, cross, tensor);
    double torSum = std::accumulate(torWeights, torWeights + nDihTrials, 0.0);
    ljWeights[lj] = hed.GetWeight() * torSum;
    uint winner = prng.PickWeighted(torWeights, nDihTrials, torSum);
    bondedEn[lj] = torEnergy[winner];
    oneFour[lj] = nonbonded_1_4[winner];
    // convert chosen torsion to 3D positions
    RotationMatrix spin =
        RotationMatrix::FromAxisAngle(-torsion[winner], cross, tensor);
    for (uint b = 0; b < hed.NumBond(); ++b) {
      // find positions
      positions[b].Set(lj, spin.Apply(positions[b][0]));
      positions[b].Add(lj, center);
    }
  }
  RestoreBranch(0);
  for (uint b = 0; b < hed.NumBond(); ++b)
    positions[b].Set(0, actualBase[b]);

  ljWeights[0] =
      OldTorsionSum(oldMol, molIndex, center, cross, tensor, prevPhi);
  ljWeights[0] *= hed.GetWeight();
  bondedEn[0] = torEnergy[0];
  oneFour[0] = nonbonded_1_4[0];

  for (uint b = 0; b < hed.NumBond(); ++b) {
    positions[b].Add(0, center);
    data->axes.WrapPBC(positions[b], oldMol.GetBox());
  }
  GOMC_INSTR(double instrB =
                 std::accumulate(ljWeights, ljWeights + nLJTrials, 0.0);)
  double stepWeight = EvalLJ(oldMol, molIndex);
  double factor = stepWeight / nLJTrials;
  if (data->lookAhead > 0 && !children.empty()) {
    // the actual trial (0) and the others each get a fresh look-ahead
    laV.resize(nLJTrials);
    LookAhead(oldMol, molIndex, nLJTrials, laV.data());
    double sel = 0.0;
    for (uint j = 0; j < nLJTrials; ++j)
      sel += ljWeights[j] * laV[j];
    // if the actual trial could not be selected, the reverse move is
    // impossible and the forward move must be rejected
    factor = laV[0] > 0.0 ? sel / (nLJTrials * laV[0]) : 1e300;
  }
#ifdef GOMC_INSTRUMENT
  if (instr::Collector::Get().On(instr::WEIGHTS)) {
    std::vector<instr::PairE> pairs;
    for (uint b = 0; b < hed.NumBond(); ++b) {
      data->calc.ParticleNonbondedPairs(oldMol, positions[b][0], hed.Bonded(b),
                                        oldMol.GetBox(), pairs);
      instr::RecordPairs(oldMol, hed.Bonded(b), pairs);
    }
  }
#endif
  for (uint b = 0; b < hed.NumBond(); ++b) {
    oldMol.ConfirmOldAtom(hed.Bonded(b));
    oldMol.AddBonds(hed.Bonded(b), hed.Focus());
  }
  oldMol.UpdateOverlap(overlap[0]);
  oldMol.AddEnergy(Energy(bondedEn[0] + hed.GetEnergy() + bondEnergy,
                          nonbonded[0] + hed.GetNonBondedEn() + oneFour[0],
                          inter[0], real[0], 0.0, 0.0, 0.0));

  GOMC_INSTR(instr::RecordOldStage(oldMol, instr::LATER, factor);)
  GOMC_INSTR(instr::RecordBondedPart(oldMol, std::log(instrB / nLJTrials));)
  oldMol.MultWeight(factor);
}

void DCLinkedHedron::LookAhead(TrialMol &mol, uint molIndex, uint nLJTrials,
                               double *v) {
  XYZArray *positions = data->multiPositions;
  for (uint j = 0; j < nLJTrials; ++j) {
    // place trial j provisionally, restoring whatever was there after
    XYZ saved[MAX_BONDS];
    bool built[MAX_BONDS];
    for (uint b = 0; b < hed.NumBond(); ++b) {
      uint a = hed.Bonded(b);
      saved[b] = mol.AtomPosition(a);
      built[b] = mol.AtomExists(a);
      mol.AddAtom(a, positions[b][j]);
    }
    v[j] = 1.0;
    for (DCLinkedHedron *c : children)
      v[j] *= c->LookAheadValue(mol, molIndex, data->lookAhead);
    for (uint b = 0; b < hed.NumBond(); ++b) {
      uint a = hed.Bonded(b);
      mol.SetAtomCoords(a, saved[b]);
      if (!built[b])
        mol.RemoveAtom(a);
    }
  }
}

double DCLinkedHedron::LookAheadValue(TrialMol &mol, uint molIndex,
                                      uint m) {
  for (uint b = 0; b < hed.NumBond(); ++b)
    if (mol.AtomExists(hed.Bonded(b)))
      return 1.0;
  const Forcefield &ff = data->ff;
  const uint nDih = data->lookAheadDih;
  // equilibrium bond lengths for both configurations: v only has to be a
  // good estimate, not the step's exact weight
  SetBondLengthNew(mol);
  hed.SetBondNew(bondLength, anchorBond);
  double prevPhi[MAX_BONDS];
  RotationMatrix cross, tensor;
  SetupBasis(mol, prevPhi, cross, tensor);
  const XYZ center = mol.AtomPosition(hed.Focus());
  for (uint b = 0; b < hed.NumBond(); ++b)
    if (laPos[b].Count() != nDih)
      laPos[b] = XYZArray(nDih);
  laTor.resize(nDih);
  laE14.resize(nDih);
  laNB.resize(nDih);

  double sum = 0.0;
  for (uint s = 0; s < m; ++s) {
    hed.PrepareLookAhead(mol, molIndex, data->lookAheadAng);
    XYZ base[MAX_BONDS];
    for (uint b = 0; b < hed.NumBond(); ++b)
      base[b] = mol.RawRectCoords(bondLength[b], hed.Theta(b), hed.Phi(b));
    for (uint t = 0; t < nDih; ++t) {
      double phi = data->prng.rand(2.0 * M_PI);
      RotationMatrix spin = RotationMatrix::FromAxisAngle(-phi, cross, tensor);
      laTor[t] = 0.0;
      laE14[t] = 0.0;
      for (uint b = 0; b < hed.NumBond(); ++b) {
        XYZ pos = spin.Apply(base[b]) + center;
        laPos[b].Set(t, pos);
        double trialPhi = hed.Phi(b) + phi;
        for (uint p = 0; p < nPrevBonds; ++p) {
          if (mol.OneFour()) {
            double distSq = mol.DistSq(pos, mol.AtomPosition(prevBonded[p]));
            laE14[t] += data->calc.IntraEnergy_1_4(distSq, prevBonded[p],
                                                   hed.Bonded(b), molIndex);
            if (std::isnan(laE14[t]))
              laE14[t] = num::BIGNUM;
          }
          laTor[t] += ff.dihedrals.Calc(dihKinds[b][p], trialPhi - prevPhi[p]);
        }
      }
    }
    std::fill(laNB.begin(), laNB.end(), 0.0);
    for (uint b = 0; b < hed.NumBond(); ++b) {
      data->axes.WrapPBC(laPos[b], mol.GetBox());
      data->calc.ParticleNonbonded(laNB.data(), mol, laPos[b], hed.Bonded(b),
                                   mol.GetBox(), nDih);
    }
    double tsum = 0.0;
    for (uint t = 0; t < nDih; ++t)
      tsum += exp(-ff.beta * (laTor[t] + laE14[t] + laNB[t]));
    sum += hed.GetWeight() * tsum / nDih;
  }
  return sum / m;
}

double DCLinkedHedron::OldTorsionSum(TrialMol &oldMol, uint molIndex,
                                     const XYZ &center, RotationMatrix &cross,
                                     RotationMatrix &tensor,
                                     const double prevPhi[]) {
  const Forcefield &ff = data->ff;
  uint nDihTrials = data->nDihTrials;
  double *torsion = data->angles;
  double *torEnergy = data->angleEnergy;
  double *nonbonded_1_4 = data->nonbonded_1_4;
  XYZArray *positions = data->multiPositions;
  double sum = 0.0;
  for (uint tor = 0; tor < nDihTrials; ++tor) {
    torsion[tor] = (tor == 0) ? 0.0 : data->prng.rand(2.0 * M_PI);
    torEnergy[tor] = 0.0;
    nonbonded_1_4[tor] = 0.0;
    for (uint b = 0; b < hed.NumBond(); ++b) {
      double trialPhi = hed.Phi(b) + torsion[tor];
      XYZ bondedC;
      if (oldMol.OneFour()) {
        // convert chosen torsion to 3D positions for bonded atoms to focus
        RotationMatrix spin =
            RotationMatrix::FromAxisAngle(-torsion[tor], cross, tensor);
        bondedC = spin.Apply(positions[b][0]) + center;
      }

      for (uint p = 0; p < nPrevBonds; ++p) {
        if (oldMol.OneFour()) {
          double distSq =
              oldMol.DistSq(bondedC, oldMol.AtomPosition(prevBonded[p]));
          nonbonded_1_4[tor] += data->calc.IntraEnergy_1_4(
              distSq, prevBonded[p], hed.Bonded(b), molIndex);
          if (std::isnan(nonbonded_1_4[tor]))
            nonbonded_1_4[tor] = num::BIGNUM;
        }
        torEnergy[tor] +=
            ff.dihedrals.Calc(dihKinds[b][p], trialPhi - prevPhi[p]);
      }
    }
    sum += exp(-ff.beta * (torEnergy[tor] + nonbonded_1_4[tor]));
  }
  return sum;
}

void DCLinkedHedron::SetupBasis(TrialMol &mol, double prevPhi[],
                                RotationMatrix &cross, RotationMatrix &tensor) {
  mol.SetBasis(hed.Focus(), hed.Prev());
  for (uint i = 0; i < nPrevBonds; ++i) {
    double th;
    // not using theta, so this is a wasted cos and sqrt
    mol.OldThetaAndPhi(prevBonded[i], hed.Prev(), th, prevPhi[i]);
  }
  XYZ rotationAxis =
      mol.AtomPosition(hed.Focus()) - mol.AtomPosition(hed.Prev());
  rotationAxis = data->axes.MinImage(rotationAxis, mol.GetBox());
  rotationAxis *= (1 / rotationAxis.Length());
  cross = RotationMatrix::CrossProduct(rotationAxis);
  tensor = RotationMatrix::TensorProduct(rotationAxis);
}

void DCLinkedHedron::CacheOldBonds(TrialMol &oldMol) {
  SetBondLengthOld(oldMol);
  bondEnergyOld = 0.0;
  for (uint i = 0; i < hed.NumBond(); ++i) {
    bondEnergyOld += data->ff.bonds.Calc(bondKinds[i], bondLengthOld[i]);
  }
}

void DCLinkedHedron::GenerateTrials(TrialMol &mol, uint molIndex, uint n,
                                    bool old, StepTrial *out) {
  PRNG &prng = data->prng;
  const Forcefield &ff = data->ff;
  uint nDihTrials = data->nDihTrials;
  double *torsion = data->angles;
  double *torWeights = data->angleWeights;
  double *torEnergy = data->angleEnergy;
  double *nonbonded_1_4 = data->nonbonded_1_4;
  XYZArray *positions = data->multiPositions;

  // Bond lengths: the old configuration's actual lengths (cached before any
  // tentative placement) or the equilibrium lengths, as in BuildOld/BuildNew
  const double *len = bondLengthOld;
  double bondE = bondEnergyOld;
  if (old) {
    hed.SetBondOld(bondLengthOld, anchorBondOld);
  } else {
    SetBondLengthNew(mol);
    hed.SetBondNew(bondLength, anchorBond);
    len = bondLength;
    bondE = 0.0;
    for (uint i = 0; i < hed.NumBond(); ++i)
      bondE += ff.bonds.Calc(bondKinds[i], bondLength[i]);
  }

  double prevPhi[MAX_BONDS];
  RotationMatrix cross, tensor;
  SetupBasis(mol, prevPhi, cross, tensor);
  const XYZ center = mol.AtomPosition(hed.Focus());

  // counting backward so positions[b][0] is free for each trial's base
  for (uint t = n; t-- > 0;) {
    if (old)
      hed.PrepareTrialOld(mol, molIndex);
    else
      hed.PrepareNew(mol, molIndex);
    for (uint b = 0; b < hed.NumBond(); ++b) {
      positions[b].Set(
          0, mol.RawRectCoords(len[b], hed.Theta(b), hed.Phi(b)));
    }
    ChooseTorsion(mol, molIndex, prevPhi, cross, tensor);
    double torSum = std::accumulate(torWeights, torWeights + nDihTrials, 0.0);
    uint winner = prng.PickWeighted(torWeights, nDihTrials, torSum);
    StepTrial &tr = out[t];
    tr.w = hed.GetWeight() * torSum; // bonded weight; completed below
    tr.torsion = torEnergy[winner];
    tr.oneFour = nonbonded_1_4[winner];
    tr.bend = hed.GetEnergy();
    tr.nb13 = hed.GetNonBondedEn();
    tr.bond = bondE;
    tr.actual = false;
    RotationMatrix spin =
        RotationMatrix::FromAxisAngle(-torsion[winner], cross, tensor);
    for (uint b = 0; b < hed.NumBond(); ++b) {
      positions[b].Set(t, spin.Apply(positions[b][0]));
      positions[b].Add(t, center);
    }
  }
  for (uint b = 0; b < hed.NumBond(); ++b) {
    data->axes.WrapPBC(positions[b], mol.GetBox());
  }
  EvalTrials(mol, molIndex, n, out);
}

void DCLinkedHedron::ActualTrial(TrialMol &oldMol, uint molIndex,
                                 StepTrial &out) {
  XYZArray *positions = data->multiPositions;
  // the old configuration's actual branch, as trial 0 in BuildOld
  hed.SetBondOld(bondLengthOld, anchorBondOld);
  hed.PrepareOld(oldMol, molIndex);
  oldMol.SetBasis(hed.Focus(), hed.Prev());
  hed.ConstrainedAnglesOld(data->nAngleTrials - 1, oldMol, molIndex);
  const XYZ center = oldMol.AtomPosition(hed.Focus());
  for (uint b = 0; b < hed.NumBond(); ++b) {
    positions[b].Set(0, oldMol.AtomPosition(hed.Bonded(b)));
    data->axes.UnwrapPBC(positions[b], 0, 1, oldMol.GetBox(), center);
    positions[b].Add(0, -center);
  }
  double prevPhi[MAX_BONDS];
  RotationMatrix cross, tensor;
  SetupBasis(oldMol, prevPhi, cross, tensor);
  double torSum =
      OldTorsionSum(oldMol, molIndex, center, cross, tensor, prevPhi);
  out.w = hed.GetWeight() * torSum;
  out.torsion = data->angleEnergy[0];
  out.oneFour = data->nonbonded_1_4[0];
  out.bend = hed.GetEnergy();
  out.nb13 = hed.GetNonBondedEn();
  out.bond = bondEnergyOld;
  out.actual = true;
  for (uint b = 0; b < hed.NumBond(); ++b) {
    positions[b].Add(0, center);
    data->axes.WrapPBC(positions[b], oldMol.GetBox());
  }
  EvalTrials(oldMol, molIndex, 1, &out);
}

// Nonbonded energies of n trials whose positions are in positions[b][0..n),
// completing w and setting p
void DCLinkedHedron::EvalTrials(TrialMol &mol, uint molIndex, uint n,
                                StepTrial *out) {
  double *inter = data->inter;
  double *real = data->real;
  double *nonbonded = data->nonbonded;
  bool *overlap = data->overlap;
  XYZArray *positions = data->multiPositions;
  std::fill_n(inter, n, 0.0);
  std::fill_n(real, n, 0.0);
  std::fill_n(nonbonded, n, 0.0);
  std::fill_n(overlap, n, false);
  // all n trials sit around the focus atom: one candidate list serves them
  data->calc.BeginLocalCandidates(mol.AtomPosition(hed.Focus()), positions,
                                  hed.NumBond(), n, mol.GetBox());
  for (uint b = 0; b < hed.NumBond(); ++b) {
    data->calc.ParticleInter(inter, real, positions[b], overlap, hed.Bonded(b),
                             molIndex, mol.GetBox(), n);
    data->calc.ParticleNonbonded(nonbonded, mol, positions[b], hed.Bonded(b),
                                 mol.GetBox(), n);
  }
  data->calc.EndLocalCandidates();
  const double beta = data->ff.beta;
  for (uint t = 0; t < n; ++t) {
    StepTrial &tr = out[t];
    for (uint b = 0; b < hed.NumBond(); ++b)
      tr.pos[b] = positions[b][t];
    tr.inter = inter[t];
    tr.real = real[t];
    tr.nonbonded = nonbonded[t];
    tr.overlap = overlap[t];
    double bu = beta * (inter[t] + real[t] + nonbonded[t]);
    tr.w *= exp(-bu); // bonded weight from the caller times exp(-beta u)
    // Trials within the threshold are always open; above it the open
    // probability falls off as exp(-beta u). An overlapping trial is never
    // opened. The old configuration's actual trial is open by construction.
    tr.p = (overlap[t] && !tr.actual)
               ? 0.0
               : std::min(1.0, exp(data->recoilThreshold - bu));
  }
}

void DCLinkedHedron::Commit(TrialMol &mol, const StepTrial &t) const {
  for (uint b = 0; b < hed.NumBond(); ++b)
    mol.AddBonds(hed.Bonded(b), hed.Focus());
  mol.UpdateOverlap(t.overlap);
  mol.AddEnergy(Energy(t.torsion + t.bend + t.bond,
                       t.nonbonded + t.nb13 + t.oneFour, t.inter, t.real, 0.0,
                       0.0, 0.0));
}

// Store the hedron's current branch as LJ trial lj
void DCLinkedHedron::SaveBranch(uint lj) {
  const uint n = hed.NumBond();
  if (trialTheta.size() < (lj + 1) * MAX_BONDS) {
    trialTheta.resize((lj + 1) * MAX_BONDS);
    trialPhi.resize((lj + 1) * MAX_BONDS);
    trialBend.resize(lj + 1);
    trialNB13.resize(lj + 1);
    trialWeight.resize(lj + 1);
  }
  for (uint b = 0; b < n; ++b) {
    trialTheta[lj * MAX_BONDS + b] = hed.Theta(b);
    trialPhi[lj * MAX_BONDS + b] = hed.Phi(b);
  }
  trialBend[lj] = hed.GetEnergy();
  trialNB13[lj] = hed.GetNonBondedEn();
  trialWeight[lj] = hed.GetWeight();
}

// Make LJ trial lj's branch the hedron's current branch
void DCLinkedHedron::RestoreBranch(uint lj) {
  hed.SetBranch(&trialTheta[lj * MAX_BONDS], &trialPhi[lj * MAX_BONDS],
                trialBend[lj], trialNB13[lj], trialWeight[lj]);
}

double DCLinkedHedron::EvalLJ(TrialMol &mol, uint molIndex) {
  uint nLJTrials = data->nLJTrialsNth;
  double *inter = data->inter;
  double *nonbonded = data->nonbonded;
  double *real = data->real;
  bool *overlap = data->overlap;
  XYZArray *positions = data->multiPositions;

  std::fill_n(data->inter, nLJTrials, 0.0);
  std::fill_n(data->nonbonded, nLJTrials, 0.0);
  std::fill_n(real, nLJTrials, 0.0);

  // all trials sit around the focus atom: one candidate list serves them
  data->calc.BeginLocalCandidates(mol.AtomPosition(hed.Focus()), positions,
                                  hed.NumBond(), nLJTrials, mol.GetBox());
  for (uint b = 0; b < hed.NumBond(); ++b) {
    data->calc.ParticleInter(inter, real, positions[b], overlap, hed.Bonded(b),
                             molIndex, mol.GetBox(), nLJTrials);

    data->calc.ParticleNonbonded(nonbonded, mol, positions[b], hed.Bonded(b),
                                 mol.GetBox(), nLJTrials);
  }
  data->calc.EndLocalCandidates();
  double stepWeight = 0;
  for (uint lj = 0; lj < nLJTrials; ++lj) {
    data->ljWeights[lj] *=
        exp(-data->ff.beta * (inter[lj] + nonbonded[lj] + real[lj]));
    stepWeight += data->ljWeights[lj];
  }
  return stepWeight;
}

void DCLinkedHedron::ChooseTorsion(TrialMol &mol, uint molIndex,
                                   double prevPhi[], RotationMatrix &cross,
                                   RotationMatrix &tensor) {
  double *torsion = data->angles;
  double *torEnergy = data->angleEnergy;
  double *torWeights = data->angleWeights;
  double *nonbonded_1_4 = data->nonbonded_1_4;
  uint nDihTrials = data->nDihTrials;
  const Forcefield &ff = data->ff;
  // To get the information if initial posotion before applying torsion
  XYZArray *positions = data->multiPositions;

  std::fill_n(torsion, data->nDihTrials, 0.0);
  std::fill_n(torWeights, data->nDihTrials, 0.0);
  std::fill_n(torEnergy, data->nDihTrials, 0.0);
  std::fill_n(nonbonded_1_4, data->nDihTrials, 0.0);

  const XYZ center = mol.AtomPosition(hed.Focus());
  // select torsion based on all dihedral angles
  for (uint tor = 0; tor < nDihTrials; ++tor) {
    torsion[tor] = data->prng.rand(2.0 * M_PI);
    torEnergy[tor] = 0.0;
    nonbonded_1_4[tor] = 0.0;
    for (uint b = 0; b < hed.NumBond(); ++b) {
      double trialPhi = hed.Phi(b) + torsion[tor];
      XYZ bondedC;
      if (mol.OneFour()) {
        // convert chosen torsion to 3D positions for bonded atoms to focus
        RotationMatrix spin =
            RotationMatrix::FromAxisAngle(-torsion[tor], cross, tensor);
        bondedC = spin.Apply(positions[b][0]) + center;
      }

      for (uint p = 0; p < nPrevBonds; ++p) {
        if (mol.OneFour()) {
          double distSq = mol.DistSq(bondedC, mol.AtomPosition(prevBonded[p]));
          nonbonded_1_4[tor] += data->calc.IntraEnergy_1_4(
              distSq, prevBonded[p], hed.Bonded(b), molIndex);
          if (std::isnan(nonbonded_1_4[tor]))
            nonbonded_1_4[tor] = num::BIGNUM;
        }

        torEnergy[tor] +=
            ff.dihedrals.Calc(dihKinds[b][p], trialPhi - prevPhi[p]);
      }
    }
    torWeights[tor] = exp(-ff.beta * (torEnergy[tor] + nonbonded_1_4[tor]));
  }
}

} // namespace cbmc
