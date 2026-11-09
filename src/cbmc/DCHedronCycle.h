/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#ifndef DCHEDRONCYCLE_H
#define DCHEDRONCYCLE_H
#include "BranchSampler.h"
#include "CBMC.h"
#include "DCComponent.h"
#include "TransformMatrix.h"

namespace mol_setup {
class MolKind;
}

namespace cbmc {
class DCData;

// Places the atoms bonded to a ring atom (focus), given the previously
// placed atom prev. Angles that belong to the ring body, or that are fixed in
// the force field, are taken from the molecule's existing geometry (bCoords);
// all other angles are drawn by BranchSampler, as in DCHedron. nAngleTrials
// geometries are drawn and one is selected with probability proportional to
// its importance weight times the 1-3 Boltzmann factor; the summed weight is
// this step's Rosenbluth factor. For the old configuration it is formed from
// the actual branch plus nAngleTrials - 1 generated ones. The constrained
// angles contribute the same factor to the new and old weights (both are
// built from the same ring geometry), so they are left out of the weight.
class DCHedronCycle {
public:
  DCHedronCycle(DCData *data, const mol_setup::MolKind &kind,
                const std::vector<int> &cycAtoms, uint focus, uint prev);
  void PrepareNew(TrialMol &newMol, uint molIndex);
  void PrepareOld(TrialMol &oldMol, uint molIndex);
  void IncorporateOld(TrialMol &oldMol, uint molIndex);
  void ConstrainedAnglesOld(uint nTrials, TrialMol &oldMol, uint molIndex);
  void SetBondNew(double const *bondLen, double const &anchBond);
  void SetBondOld(double const *bondLen, double const &anchBond);
  uint Bonded(uint i) const { return bonded[i]; }
  double Theta(uint i) const { return theta[i]; }
  double Phi(uint i) const { return phi[i]; }
  double GetWeight() const { return weight; }
  double GetEnergy() const { return bendEnergy; }
  double GetNonBondedEn() const { return oneThree; }
  uint NumBond() const { return nBonds; }
  uint Focus() const { return focus; }
  uint Prev() const { return prev; }
  //! Record this step's 1-3 pairs as counted (after the step is placed)
  void MarkCounted13(TrialMol &mol) const;

  // need to go to private
  uint bonded[MAX_BONDS];

private:
  void BranchFailure(TrialMol &mol) const;
  // Read the values of the constrained angles, and the mirror solution of
  // each constrained pair, from the existing geometry
  void SetConstraints(TrialMol &mol);
  void SampleTrials(TrialMol &mol, uint molIndex, bool useOld, uint nTrials);
  void BranchEnergy(TrialMol &mol, uint molIndex, const double *th,
                    const double *ph, bool useOld, double &bend,
                    double &nb13);
  // to calculate the angle in the ring from bCoords or tCoords a0-a1-a2
  double CalcTheta(TrialMol &mol, const uint a0, const uint a1, const uint a2);
  //! Sets an orthonormal basis for coordinate conversion.
  /*!\param p1 Index of focus
   * \param p2 Index of prev
   */
  void SetBasis(TrialMol &mol, uint p1, uint p2);
  //! Calculates theta and phi coords for atom in the current basis
  //! centered on lastAtom. phi in (-pi, pi]
  double CalcOldPhi(TrialMol &mol, uint atom, uint lastAtom) const;

  DCData *data;
  uint focus, prev;
  uint nBonds;
  // atoms bonded to focus, being build

  // bond length of atom bonded to focus
  double bondLength[MAX_BONDS];
  double bondLengthOld[MAX_BONDS];

  // angleKinds[i][j] = kind between bonded[i] and bonded[j]
  // except angleKinds[i][i] = kind between bonded[i] and prev
  uint angleKinds[MAX_BONDS][MAX_BONDS];
  bool angleInRing[MAX_BONDS][MAX_BONDS];
  double theta[MAX_BONDS];
  double phi[MAX_BONDS];
  double weight;
  double bendEnergy, oneThree;
  double anchorBond, anchorBondOld;
  BranchSampler sampler;
  RotationMatrix growthToWorld;
  RotationMatrix worldToGrowth;
};
} // namespace cbmc

#endif /*DCHEDRONCYCLE_H*/
