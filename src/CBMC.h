/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#ifndef CBMC_H
#define CBMC_H

/*    CBMC.h
 *     Base Class for CBMC algorithms
 *     Also includes Factory Functions for same
 *
 */

#include <vector>

#include "BasicTypes.h"
#include "Forcefield.h"

struct MolPick;
class Forcefield;
class MoleculeKind;
class Setup;
class System;

namespace cbmc {
class TrialMol;

class CBMC {
public:
  // Builds a new molecule using a CBMC algorithm, oldMol and newMol
  // will be modified to contain the energies of the old and new sites
  virtual void Build(TrialMol &oldMol, TrialMol &newMol, uint molIndex) = 0;

  // Regrowing the molecule using a CBMC algorithm, oldMol and newMol
  virtual void Regrowth(TrialMol &oldMol, TrialMol &newMol, uint molIndex) = 0;

  // Rotate the atoms between two nodes arounde  the vector that connects two
  // nodes using crank shaft algorithm, oldMol and newMol
  virtual void CrankShaft(TrialMol &oldMol, TrialMol &newMol,
                          uint molIndex) = 0;

  // Rigid insertion of molecule and perform position and rotational trial
  virtual void BuildIDNew(TrialMol &newMol, uint molIndex) = 0;
  virtual void BuildIDOld(TrialMol &oldMol, uint molIndex) = 0;

  // Build the molecule using CD-CBMC
  virtual void BuildNew(TrialMol &newMol, uint molIndex) = 0;
  virtual void BuildOld(TrialMol &oldMol, uint molIndex) = 0;

  // Grow the molecule from predefined atom (node)
  virtual void BuildGrowNew(TrialMol &newMol, uint molIndex) = 0;
  virtual void BuildGrowOld(TrialMol &oldMol, uint molIndex) = 0;

  // grow the molecule with starting seed within subVolume
  virtual void BuildGrowInCav(TrialMol &oldMol, TrialMol &newMol,
                              uint molIndex) = 0;

#ifdef GOMC_INSTRUMENT
  // SMC probe (instrumentation, see DCGraph::ProbeSMC). For each (N, k)
  // setting, out receives ln Z_hat of an SMC insertion into newMol's box,
  // ln Z_hat of a conditional SMC weighing of oldMol and the wall time of
  // the pair in seconds, in that order. Returns
  // false if this builder does not support it.
  struct SMCProbeSetting {
    uint N, k;
  };
  virtual bool ProbeSMC(TrialMol &oldMol, TrialMol &newMol, uint molIndex,
                        const std::vector<SMCProbeSetting> &settings,
                        std::vector<double> &out) {
    return false;
  }
#endif

  virtual ~CBMC() {}
};

// Max allowed bonds to any atom
static const uint MAX_BONDS = 6;
// Factory function, determines, prepares and returns appropriate CBMC
CBMC *MakeCBMC(System &sys, const Forcefield &ff, const MoleculeKind &kind,
               const Setup &set);
} // namespace cbmc

#endif /*CBMC_H*/
