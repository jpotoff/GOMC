/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#ifndef DCLINKEDHEDRON_H
#define DCLINKEDHEDRON_H
#include <vector>

#include "CBMC.h"
#include "DCComponent.h"
#include "DCHedron.h"
#include "TransformMatrix.h"

namespace mol_setup {
class MolKind;
}

namespace cbmc {
class DCData;
class DCLinkedHedron : public DCComponent {
public:
  DCLinkedHedron(DCData *data, const mol_setup::MolKind &kind, uint focus,
                 uint prev);
  void PrepareNew(TrialMol &newMol, uint molIndex);
  void PrepareOld(TrialMol &oldMol, uint molIndex);
  void BuildOld(TrialMol &oldMol, uint molIndex);
  void BuildNew(TrialMol &newMol, uint molIndex);
  void SetBondLengthNew(TrialMol &newMol);
  void SetBondLengthOld(TrialMol &oldMol);

  DCComponent *Clone() { return new DCLinkedHedron(*this); };

  // Trial interface for recoil growth (RecoilGrowth.h, used by DCGraph).
  // One trial is what one LJ trial is in BuildNew: a fresh branch, a torsion
  // chosen from CBMC_Dih trials, and its nonbonded energy. Trials are
  // generated against the atoms placed in mol; nothing is committed.
  struct StepTrial {
    XYZ pos[MAX_BONDS]; // positions of the atoms bonded to focus
    double w;           // full weight: bonded weight * exp(-beta u)
    double p;           // open probability min(1, exp(threshold - beta u))
    double torsion, oneFour, bend, nb13, bond; // bonded energies
    double inter, real, nonbonded;             // u = inter + real + nonbonded
    bool overlap;
    bool actual; // the old configuration's actual placement
  };
  // Old bond lengths from the actual old configuration. Call before any
  // tentative placement changes oldMol's coordinates.
  void CacheOldBonds(TrialMol &oldMol);
  void GenerateTrials(TrialMol &mol, uint molIndex, uint n, bool old,
                      StepTrial *out);
  void ActualTrial(TrialMol &oldMol, uint molIndex, StepTrial &out);
  // Add the bonds and energies of a trial whose atoms are placed in mol.
  void Commit(TrialMol &mol, const StepTrial &t) const;
  uint NumBonded() const { return hed.NumBond(); }
  uint BondedAtom(uint b) const { return hed.Bonded(b); }

  // One-step look-ahead (CBMC_LookAhead). The children are the steps that
  // grow from the atoms this step places; DCGraph links them.
  void SetChildren(const std::vector<DCLinkedHedron *> &c) { children = c; }
  // Estimate of this step's weight given the atoms placed in mol: the mean
  // over m branch draws of the branch weight times the mean torsion weight,
  // with torsion, 1-4 and 1-5+ energies. Leaves mol unchanged apart from
  // its growth basis. Returns 1 if any atom of this step is already built.
  double LookAheadValue(TrialMol &mol, uint molIndex, uint m);

private:
  void ChooseTorsion(TrialMol &mol, uint molIndex, double prevPhi[],
                     RotationMatrix &cross, RotationMatrix &tensor);
  double EvalLJ(TrialMol &mol, uint molIndex);
  // Sum of torsion weights for the old configuration's actual branch, with
  // the actual torsion as trial 0 (positions[b][0] hold the branch atoms
  // relative to center)
  double OldTorsionSum(TrialMol &oldMol, uint molIndex, const XYZ &center,
                       RotationMatrix &cross, RotationMatrix &tensor,
                       const double prevPhi[]);
  void SetupBasis(TrialMol &mol, double prevPhi[], RotationMatrix &cross,
                  RotationMatrix &tensor);
  void EvalTrials(TrialMol &mol, uint molIndex, uint n, StepTrial *out);
  // Look-ahead values v_j of this step's LJ trials (positions in
  // data->multiPositions), each with trial j placed provisionally
  void LookAhead(TrialMol &mol, uint molIndex, uint nLJTrials, double *v);
  void SaveBranch(uint lj);
  void RestoreBranch(uint lj);
  DCData *data;
  DCHedron hed;
  uint nPrevBonds;
  uint prevBonded[MAX_BONDS];
  // kind[bonded][previous]
  uint dihKinds[MAX_BONDS][MAX_BONDS];

  // bond energy of built branch
  double bondEnergy;
  // bond energy of the old configuration (recoil growth)
  double bondEnergyOld;
  // bond length of prev bonded to focus
  double anchorBond, anchorBondOld;
  // bond length of atom bonded to focus
  double bondLength[MAX_BONDS];
  double bondLengthOld[MAX_BONDS];
  // bondKind between bonded[i] and focus
  uint bondKinds[MAX_BONDS];
  // branch geometry, energies and weight of each LJ trial
  std::vector<double> trialTheta, trialPhi; // [trial * MAX_BONDS + bond]
  std::vector<double> trialBend, trialNB13, trialWeight;
  // one-step look-ahead (CBMC_LookAhead)
  std::vector<DCLinkedHedron *> children;
  std::vector<double> laV, laSel, laTor, laE14, laNB; // scratch
  XYZArray laPos[MAX_BONDS];
};
} // namespace cbmc
#endif /*DCLINKEDHEDRON_H*/
