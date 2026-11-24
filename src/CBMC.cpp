/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#include "CBMC.h"

#include <set>
#include <string>
#include <vector>

#include "DCCyclic.h"
#include "DCGraph.h"
#include "DCLinear.h"
#include "FlexibleBond.h"
#include "Forcefield.h"
#include "MolSetup.h"
#include "MoleculeKind.h"

namespace cbmc {

CBMC *MakeCBMC(System &sys, const Forcefield &ff, const MoleculeKind &kind,
               const Setup &set) {
  std::vector<uint> bondCount(kind.NumAtoms(), 0);
  for (uint i = 0; i < kind.bondList.count; ++i) {
    bondCount[kind.bondList.part1[i]]++;
    bondCount[kind.bondList.part2[i]]++;
  }

  bool cyclic = (kind.NumBonds() > kind.NumAtoms() - 1) ? true : false;

  // kinds can be set up more than once per run; report each once
  static std::set<std::string> reported;
  if (set.config.sys.cbmcTrials.flexibleBonds && kind.NumBonds() > 0 &&
      reported.insert(kind.name).second) {
    // report which bonds of this kind CBMC samples (FlexibleBond.h)
    mol_setup::MolMap::const_iterator it =
        set.mol.kindMap.find(kind.uniqueName);
    uint nFlex = 0, nHeavy = 0;
    if (it != set.mol.kindMap.end()) {
      const mol_setup::MolKind &mk = it->second;
      for (const mol_setup::Bond &b : mk.bonds) {
        if (mk.atoms[b.a0].mass >= FLEXIBLE_BOND_MIN_MASS &&
            mk.atoms[b.a1].mass >= FLEXIBLE_BOND_MIN_MASS)
          ++nHeavy;
        if (!cyclic && IsFlexibleBond(true, ff.bonds, mk, b.a0, b.a1, b.kind))
          ++nFlex;
      }
    }
    std::string label = "Info: Flexible bonds, " + kind.name;
    char msg[256];
    if (cyclic)
      snprintf(msg, sizeof(msg),
               "0 of %u bonds sampled (ring molecule: all bonds fixed)",
               kind.NumBonds());
    else
      snprintf(msg, sizeof(msg),
               "%u of %u bonds sampled (%u heavy-atom bonds; %u with H fixed)",
               nFlex, kind.NumBonds(), nHeavy, kind.NumBonds() - nHeavy);
    printf("%-40s %-s \n", label.c_str(), msg);
  }

  if (cyclic) {
    return new DCCyclic(sys, ff, kind, set);
  } else if (kind.NumAtoms() > 2) {
    // Any molecule woth 3 atoms and more will be built in DCGraph
    return new DCGraph(sys, ff, kind, set);
  } else {
    return new DCLinear(sys, ff, kind, set);
  }
}

} // namespace cbmc
