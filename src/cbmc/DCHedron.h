/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#ifndef DCHEDRON_H
#define DCHEDRON_H
#include "CBMC.h"
#include "DCComponent.h"

namespace mol_setup {
class MolKind;
}

namespace cbmc {
class DCData;
struct BendTable;

// Places the atoms bonded to a focus atom (the "branch"), given the
// previously placed atom prev. All branch directions are sampled jointly and
// exactly from the bending Boltzmann distribution
//   p ~ prod_b sin(theta_b) exp(-beta U(prev,focus,b))
//       * prod_{c<b} exp(-beta U(c,focus,b)),
// so the bending contribution to the Rosenbluth weight is a constant that
// cancels between new and old configurations. If 1-3 nonbonded interactions
// are enabled they are Rosenbluth weighted over nAngleTrials exact samples.
class DCHedron {
public:
  DCHedron(DCData *data, const mol_setup::MolKind &kind, uint focus, uint prev);
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

  // need to go to private
  uint bonded[MAX_BONDS];

private:
  const BendTable &GetTable(uint kind);
  double SampleTheta(uint kind);
  void SampleBranch(TrialMol &mol, double *th, double *ph);
  void BranchEnergy(TrialMol &mol, uint molIndex, const double *th,
                    const double *ph, bool useOld, double &bend,
                    double &nb13);

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
  double theta[MAX_BONDS];
  double phi[MAX_BONDS];
  double weight;
  double bendEnergy, oneThree;
  double anchorBond, anchorBondOld;
};
} // namespace cbmc

#endif /*DCHEDRON_H*/
