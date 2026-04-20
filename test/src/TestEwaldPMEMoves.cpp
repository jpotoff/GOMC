#include "CalculateEnergy.h"
#include "Coordinates.h"
#include "EwaldPME.h"
#include "Molecules.h"
#include "Simulation.h"
#include "cbmc/TrialMol.h"
#include "gtest/gtest.h"
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <unistd.h>

class EwaldPMEMovesTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create a temporary directory and copy files
    if (system("mkdir -p /tmp/gomc_pme_repro")) {
    }
    if (system("cp "
               "/home/ai8111/recovery/GOMC/test/input/Systems/OPC/Base/"
               "OPC_FF.inp /tmp/gomc_pme_repro/")) {
    }
    if (system(
            "cp /home/ai8111/recovery/GOMC/test/input/Systems/OPC/Base/*.pdb "
            "/tmp/gomc_pme_repro/")) {
    }
    if (system(
            "cp /home/ai8111/recovery/GOMC/test/input/Systems/OPC/Base/*.psf "
            "/tmp/gomc_pme_repro/")) {
    }

    // Create in.conf
    FILE *f = fopen("/tmp/gomc_pme_repro/in.conf", "w");
    fprintf(f, "Restart False\n");
    fprintf(f, "Checkpoint False\n");
    fprintf(f, "ExpertMode False\n");
    fprintf(f, "PRNG INTSEED\n");
    fprintf(f, "Random_Seed 12345\n");
    fprintf(f, "ParaTypeCHARMM True\n");
    fprintf(f, "Parameters ./OPC_FF.inp\n");
    fprintf(f, "Coordinates 0 ./initial_box_0.pdb\n");
    fprintf(f, "Structure 0 ./initial_box_0.psf\n");
    fprintf(f, "Temperature 300.0\n");
    fprintf(f, "Pressure 1.0\n");
    fprintf(f, "useConstantArea False\n");
    fprintf(f, "Potential VDW\n");
    fprintf(f, "LRC True\n");
    fprintf(f, "IPC False\n");
    fprintf(f, "Rcut 8.0\n");
    fprintf(f, "RcutLow 0.7\n");
    fprintf(f, "Exclude 1-4\n");
    fprintf(f, "VDWGeometricSigma False\n");
    fprintf(f, "ElectrostaticMethod PME\n");
    fprintf(f, "ElectroStatic True\n");
    fprintf(f, "Tolerance 1e-05\n");
    fprintf(f, "1-4scaling 0.0\n");
    fprintf(f, "PressureCalc False 10000000\n");
    fprintf(f, "PMESplineOrder 4\n");
    fprintf(f, "PMEGridSpacing 1.5\n");
    fprintf(f, "PMERefreshFreq 100\n");
    fprintf(f, "RunSteps 2\n");
    fprintf(f, "EqSteps 1\n");
    fprintf(f, "AdjSteps 1\n");
    fprintf(f, "DisFreq 0.49\n");
    fprintf(f, "RotFreq 0.49\n");
    fprintf(f, "VolFreq 0.02\n");
    fprintf(f, "CellBasisVector1 0 25.0 0.0 0.0\n");
    fprintf(f, "CellBasisVector2 0 0.0 25.0 0.0\n");
    fprintf(f, "CellBasisVector3 0 0.0 0.0 25.0\n");
    fprintf(f, "CBMC_First 12\n");
    fprintf(f, "CBMC_Nth 10\n");
    fprintf(f, "CBMC_Ang 50\n");
    fprintf(f, "CBMC_Dih 50\n");
    fprintf(f, "OutputName repro\n");
    fprintf(f, "RestartFreq True 10000000\n");
    fprintf(f, "CheckpointFreq True 10000000\n");
    fprintf(f, "CoordinatesFreq False 10000000\n");
    fprintf(f, "DCDFreq True 10000000\n");
    fprintf(f, "ConsoleFreq True 1\n");
    fprintf(f, "BlockAverageFreq True 10000000\n");
    fprintf(f, "HistogramFreq False 10000000\n")
        fprintf(f, "OutEnergy True True\n");
    fprintf(f, "OutPressure True True\n");
    fprintf(f, "OutMolNum True True\n");
    fprintf(f, "OutDensity True True\n");
    fprintf(f, "OutVolume True True\n");
    fprintf(f, "OutSurfaceTension False False\n");
    fclose(f);

    origDir = getcwd(NULL, 0);
    if (chdir("/tmp/gomc_pme_repro")) {
    }
  }

  void TearDown() override {
    if (origDir) {
      if (chdir(origDir)) {
      }
      free(origDir);
    }
    // if(system("rm -rf /tmp/gomc_pme_repro")) {}
  }
  char *origDir;
};

TEST_F(EwaldPMEMovesTest, DisplacementConsistency) {
  Simulation sim("in.conf");
  EwaldPME *pme = dynamic_cast<EwaldPME *>(sim.GetEwald());
  ASSERT_NE(pme, nullptr);

  uint box = 0;
  // Calculate initial energy
  sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();
  double initialEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;
  std::cout << "Initial Reciprocal Energy: " << initialEnergy << " K"
            << std::endl;

  // Pick a molecule to move
  uint molIndex = 10;
  if (molIndex >= sim.GetMolecules().count)
    molIndex = 0;

  XYZ move(1.5, -0.8, 2.1); // Arbitrary displacement

  // Get current coords for the molecule
  uint nAtoms = sim.GetMolecules().GetKind(molIndex).NumAtoms();
  XYZArray newCoords(nAtoms);
  uint startAtom = sim.GetMolecules().MolStart(molIndex);
  for (uint i = 0; i < nAtoms; ++i) {
    newCoords.Set(i, sim.GetCoordinates().Get(startAtom + i) + move);
  }

  // Calculate dE incrementally
  double dE = pme->MolReciprocal(newCoords, molIndex, box);
  std::cout << "Incremental dE reported by EwaldPME: " << dE << " K"
            << std::endl;

  // Update state in PME (move accepted)
  pme->UpdateRecip(box);

  // In GOMC with PME, the energy tracker is now directly updated to the
  // exact reciprocal energy within UpdateRecip.
  double expectedEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;
  std::cout << "Expected Total Reciprocal Energy (from exact tracker): "
            << expectedEnergy << " K" << std::endl;

  // Manually update the coordinates in the simulation for the full sum check
  for (uint i = 0; i < nAtoms; ++i) {
    sim.GetCoordinates().Set(startAtom + i, newCoords.Get(i));
  }

  // Now perform a full reciprocal sum to verify consistency
  pme->UpdateVectorsAndRecipTerms(false);
  sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();
  double actualEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;
  std::cout << "Actual Full Sum Reciprocal Energy: " << actualEnergy << " K"
            << std::endl;

  // If there is a massive bug, this will fail spectacularly
  EXPECT_NEAR(expectedEnergy, actualEnergy, 1e-1);

  // Also check if actualEnergy is reasonable (not -1.5E6)
  EXPECT_GT(actualEnergy, -1e5);
}

TEST_F(EwaldPMEMovesTest, RotationMoveConsistency) {
  Simulation sim("in.conf");
  EwaldPME *pme = dynamic_cast<EwaldPME *>(sim.GetEwald());
  ASSERT_NE(pme, nullptr);

  uint box = 0;
  // Calculate initial energy
  sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();
  double initialEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;
  std::cout << "Initial Reciprocal Energy: " << initialEnergy << " K"
            << std::endl;

  // Pick a molecule to move
  uint molIndex = 11;
  if (molIndex >= sim.GetMolecules().count)
    molIndex = 0;

  // Get current coords for the molecule
  uint nAtoms = sim.GetMolecules().GetKind(molIndex).NumAtoms();
  XYZArray newCoords(nAtoms);
  uint startAtom = sim.GetMolecules().MolStart(molIndex);

  // Calculate COM (or just use atom 0 as pivot)
  XYZ pivot = sim.GetCoordinates().Get(startAtom);

  // Create an arbitrary rotation matrix (e.g., ~90 deg around Z axis)
  double angle = 1.57079632679; // pi/2
  double c = std::cos(angle);
  double s = std::sin(angle);

  for (uint i = 0; i < nAtoms; ++i) {
    XYZ pos = sim.GetCoordinates().Get(startAtom + i);
    // Translate to origin
    pos.x -= pivot.x;
    pos.y -= pivot.y;
    pos.z -= pivot.z;

    // Rotate around Z axis
    double rx = pos.x * c - pos.y * s;
    double ry = pos.x * s + pos.y * c;
    double rz = pos.z;

    // Translate back
    pos.x = rx + pivot.x;
    pos.y = ry + pivot.y;
    pos.z = rz + pivot.z;

    newCoords.Set(i, pos);
  }

  // Calculate dE incrementally
  double dE = pme->MolReciprocal(newCoords, molIndex, box);
  std::cout << "Incremental dE reported by EwaldPME: " << dE << " K"
            << std::endl;

  // Update state in PME (move accepted)
  pme->UpdateRecip(box);

  // In GOMC with PME, the energy tracker is now directly updated to the
  // exact reciprocal energy within UpdateRecip.
  double expectedEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;
  std::cout << "Expected Total Reciprocal Energy (from exact tracker): "
            << expectedEnergy << " K" << std::endl;

  // Manually update the coordinates in the simulation for the full sum check
  for (uint i = 0; i < nAtoms; ++i) {
    sim.GetCoordinates().Set(startAtom + i, newCoords.Get(i));
  }

  // Now perform a full reciprocal sum to verify consistency
  pme->UpdateVectorsAndRecipTerms(false);
  sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();
  double actualEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;
  std::cout << "Actual Full Sum Reciprocal Energy: " << actualEnergy << " K"
            << std::endl;

  // If there is a massive bug, this will fail spectacularly
  EXPECT_NEAR(expectedEnergy, actualEnergy, 1e-1);

  // Also check if actualEnergy is reasonable (not -1.5E6)
  EXPECT_GT(actualEnergy, -1e5);
}

TEST_F(EwaldPMEMovesTest, VolumeMoveConsistency) {
  Simulation sim("in.conf");
  EwaldPME *pme = dynamic_cast<EwaldPME *>(sim.GetEwald());
  ASSERT_NE(pme, nullptr);

  uint box = 0;
  // Calculate initial energy
  sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();
  double initialEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;
  std::cout << "Initial Reciprocal Energy: " << initialEnergy << " K"
            << std::endl;

  // Manually shift the box dims
  BoxDimensions newAxes = sim.GetBoxDim();
  XYZ scale(1.05, 1.05, 1.05);
  newAxes.axis.Set(0, XYZ(newAxes.axis.Get(0).x * 1.05,
                          newAxes.axis.Get(0).y * 1.05,
                          newAxes.axis.Get(0).z * 1.05));
  newAxes.halfAx.Set(0, XYZ(newAxes.halfAx.Get(0).x * 1.05,
                            newAxes.halfAx.Get(0).y * 1.05,
                            newAxes.halfAx.Get(0).z * 1.05));
  newAxes.volume[0] *= (1.05 * 1.05 * 1.05);
  newAxes.volInv[0] = 1.0 / newAxes.volume[0];
  newAxes.cellBasis[0].Set(0, XYZ(newAxes.axis.Get(0).x, 0.0, 0.0));
  newAxes.cellBasis[0].Set(1, XYZ(0.0, newAxes.axis.Get(0).y, 0.0));
  newAxes.cellBasis[0].Set(2, XYZ(0.0, 0.0, newAxes.axis.Get(0).z));

  // Scale coords
  XYZArray newCoords(sim.GetCoordinates().Count());
  for (uint i = 0; i < newCoords.Count(); ++i) {
    XYZ pos = sim.GetCoordinates().Get(i);
    // Origin is implicitly 0,0,0
    pos.x *= 1.05;
    pos.y *= 1.05;
    pos.z *= 1.05;
    newCoords.Set(i, pos);
  }

  // Calculate trial energy
  pme->RecipInit(box, newAxes);
  pme->BoxReciprocalSetup(box, newCoords);
  double trialEnergy = pme->BoxReciprocal(box, true);
  std::cout << "Incremental dE reported by EwaldPME: "
            << (trialEnergy - initialEnergy) << " K" << std::endl;

  // Accept move
  sim.GetBoxDim() = newAxes;
  for (uint i = 0; i < newCoords.Count(); ++i) {
    sim.GetCoordinates().Set(i, newCoords.Get(i));
  }
  pme->UpdateRecip(box);
  pme->UpdateRecipVec(box);

  sim.GetSystemEnergy().boxEnergy[box].recip = trialEnergy;
  double expectedEnergy = trialEnergy;
  std::cout << "Expected Total Reciprocal Energy: " << expectedEnergy << " K"
            << std::endl;

  // Now perform a full reciprocal sum to verify consistency
  pme->UpdateVectorsAndRecipTerms(false);
  sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();
  double actualEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;
  std::cout << "Actual Full Sum Reciprocal Energy: " << actualEnergy << " K"
            << std::endl;

  // If there is a massive bug, this will fail spectacularly
  EXPECT_NEAR(expectedEnergy, actualEnergy, 1e-1);

  // Also check if actualEnergy is reasonable
  EXPECT_GT(actualEnergy, -1e5);
}

TEST_F(EwaldPMEMovesTest, IntraSwapMoveConsistency) {
  Simulation sim("in.conf");
  EwaldPME *pme = dynamic_cast<EwaldPME *>(sim.GetEwald());
  ASSERT_NE(pme, nullptr);

  uint box = 0;
  // Calculate initial energy
  sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();
  double initialEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;
  std::cout << "Initial Reciprocal Energy: " << initialEnergy << " K"
            << std::endl;

  // Pick a molecule to modify internally
  uint molIndex = 12;
  if (molIndex >= sim.GetMolecules().count)
    molIndex = 0;

  // Get current coords for the molecule
  uint nAtoms = sim.GetMolecules().GetKind(molIndex).NumAtoms();
  if (nAtoms < 2) {
    GTEST_SKIP() << "Molecule has less than 2 atoms, cannot perform IntraSwap";
  }

  XYZArray newCoords(nAtoms);
  uint startAtom = sim.GetMolecules().MolStart(molIndex);
  for (uint i = 0; i < nAtoms; ++i) {
    newCoords.Set(i, sim.GetCoordinates().Get(startAtom + i));
  }

  // Perform an intramolecular swap (exchange positions of atom 0 and atom 1)
  XYZ temp = newCoords.Get(0);
  newCoords.Set(0, newCoords.Get(1));
  newCoords.Set(1, temp);

  // Calculate dE incrementally
  double dE = pme->MolReciprocal(newCoords, molIndex, box);
  std::cout << "Incremental dE reported by EwaldPME: " << dE << " K"
            << std::endl;

  // Update state in PME (move accepted)
  pme->UpdateRecip(box);

  // In GOMC, the energy tracker is now updated exactly by UpdateRecip.
  double expectedEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;
  std::cout << "Expected Total Reciprocal Energy (from exact tracker): "
            << expectedEnergy << " K" << std::endl;

  // Manually update the coordinates in the simulation for the full sum check
  for (uint i = 0; i < nAtoms; ++i) {
    sim.GetCoordinates().Set(startAtom + i, newCoords.Get(i));
  }

  // Now perform a full reciprocal sum to verify consistency
  pme->UpdateVectorsAndRecipTerms(false);
  sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();
  double actualEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;
  std::cout << "Actual Full Sum Reciprocal Energy: " << actualEnergy << " K"
            << std::endl;

  // If there is a massive bug, this will fail spectacularly
  EXPECT_NEAR(expectedEnergy, actualEnergy, 1e-1);

  // Also check if actualEnergy is reasonable (not -1.5E6)
  EXPECT_GT(actualEnergy, -1e5);
}

TEST_F(EwaldPMEMovesTest, SwapMoveConsistency) {
  Simulation sim("in.conf");
  EwaldPME *pme = dynamic_cast<EwaldPME *>(sim.GetEwald());
  ASSERT_NE(pme, nullptr);

  uint box = 0;
  // Calculate initial energy
  sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();
  double initialEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;
  std::cout << "Initial Reciprocal Energy: " << initialEnergy << " K"
            << std::endl;

  // Pick a molecule to delete and re-insert
  uint molIndex = 14;
  if (molIndex >= sim.GetMolecules().count)
    molIndex = 0;

  MoleculeKind const &kind = sim.GetMolecules().GetKind(molIndex);
  uint nAtoms = kind.NumAtoms();
  uint startAtom = sim.GetMolecules().MolStart(molIndex);

  // 1. Build "Old" TrialMol matching current coordinates
  cbmc::TrialMol oldMol(kind, sim.GetBoxDim(), box);
  oldMol.SetCoords(sim.GetCoordinates(), startAtom);

  // 2. Build "New" TrialMol shifted by an arbitrary vector
  cbmc::TrialMol newMol(kind, sim.GetBoxDim(), box);
  XYZArray shiftedCoords(sim.GetCoordinates().Count());
  for (uint i = 0; i < shiftedCoords.Count(); ++i) {
    shiftedCoords.Set(i, sim.GetCoordinates().Get(i));
  }

  XYZ move(4.2, -3.1, 1.9); // Arbitrary insertion site delta
  for (uint i = 0; i < nAtoms; ++i) {
    shiftedCoords.Set(startAtom + i, shiftedCoords.Get(startAtom + i) + move);
  }
  newMol.SetCoords(shiftedCoords, startAtom);

  // 3. Test SwapSourceRecip (Deletion)
  double dE_del = pme->SwapSourceRecip(oldMol, box, molIndex);
  std::cout << "Incremental dE_del reported by PME deletion: " << dE_del << " K"
            << std::endl;
  pme->UpdateRecip(box); // Accept Deletion

  // 4. Test SwapDestRecip (Insertion)
  double dE_ins = pme->SwapDestRecip(newMol, box, molIndex);
  std::cout << "Incremental dE_ins reported by PME insertion: " << dE_ins
            << " K" << std::endl;
  pme->UpdateRecip(box); // Accept Insertion

  // Net expected tracking
  double expectedEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;

  std::cout << "Expected Total Reciprocal Energy (from exact tracker): "
            << expectedEnergy << " K" << std::endl;

  // 5. Update coordinates array so SystemTotal can build it from scratch
  for (uint i = 0; i < nAtoms; ++i) {
    sim.GetCoordinates().Set(startAtom + i, shiftedCoords.Get(startAtom + i));
  }

  pme->UpdateVectorsAndRecipTerms(false);
  sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();
  double actualEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;
  std::cout << "Actual Full Sum Reciprocal Energy: " << actualEnergy << " K"
            << std::endl;

  // Verify consistency!
  EXPECT_NEAR(expectedEnergy, actualEnergy, 1e-1);
  EXPECT_GT(actualEnergy, -1e5);
}

TEST_F(EwaldPMEMovesTest, CombinedVolumeAndDisplacementMoveConsistency) {
  Simulation sim("in.conf");
  EwaldPME *pme = dynamic_cast<EwaldPME *>(sim.GetEwald());
  ASSERT_NE(pme, nullptr);

  uint box = 0;
  sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();
  // We use this just to get past initialization

  // -- 1. VOLUME MOVE (REJECTED) --
  // We use a large scale to force K to change (25 * 1.2 = 30; 30/1.5 = 20 > 18)
  BoxDimensions newAxes = sim.GetBoxDim();
  newAxes.axis.Set(0, XYZ(newAxes.axis.Get(0).x * 1.20,
                          newAxes.axis.Get(0).y * 1.20,
                          newAxes.axis.Get(0).z * 1.20));
  newAxes.halfAx.Set(0, XYZ(newAxes.halfAx.Get(0).x * 1.20,
                            newAxes.halfAx.Get(0).y * 1.20,
                            newAxes.halfAx.Get(0).z * 1.20));
  newAxes.volume[0] *= (1.20 * 1.20 * 1.20);
  newAxes.volInv[0] = 1.0 / newAxes.volume[0];
  newAxes.cellBasis[0].Set(0, XYZ(newAxes.axis.Get(0).x, 0.0, 0.0));
  newAxes.cellBasis[0].Set(1, XYZ(0.0, newAxes.axis.Get(0).y, 0.0));
  newAxes.cellBasis[0].Set(2, XYZ(0.0, 0.0, newAxes.axis.Get(0).z));

  XYZArray newCoords(sim.GetCoordinates().Count());
  for (uint i = 0; i < newCoords.Count(); ++i) {
    XYZ pos = sim.GetCoordinates().Get(i);
    pos.x *= 1.20;
    pos.y *= 1.20;
    pos.z *= 1.20;
    newCoords.Set(i, pos);
  }

  pme->RecipInit(box, newAxes);
  pme->BoxReciprocalSetup(box, newCoords);
  double trialEnergy = pme->BoxReciprocal(box, true);

  // REJECT Volume Move
  // Do not accept new volume / coords.
  // Call exgMolCache to simulate what GOMC does on rejection
  pme->exgMolCache();

  // -- 2. DISPLACEMENT MOVE --
  uint molIndex = 10;
  if (molIndex >= sim.GetMolecules().count)
    molIndex = 0;

  XYZ move(1.5, -0.8, 2.1);
  uint nAtoms = sim.GetMolecules().GetKind(molIndex).NumAtoms();
  XYZArray dispCoords(nAtoms);
  uint startAtom = sim.GetMolecules().MolStart(molIndex);
  for (uint i = 0; i < nAtoms; ++i) {
    dispCoords.Set(i, sim.GetCoordinates().Get(startAtom + i) + move);
  }

  double dE = pme->MolReciprocal(dispCoords, molIndex, box);

  // Accept Displacement
  pme->UpdateRecip(box);
  double expectedEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;

  for (uint i = 0; i < nAtoms; ++i) {
    sim.GetCoordinates().Set(startAtom + i, dispCoords.Get(i));
  }

  // Verify
  pme->UpdateVectorsAndRecipTerms(false);
  sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();
  double actualEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;

  std::cout << "Expected: " << expectedEnergy << " Actual: " << actualEnergy
            << std::endl;
  EXPECT_NEAR(expectedEnergy, actualEnergy, 1e-1);
}

// ---------------------------------------------------------------------------
// Diagnostic: scan raw PME BoxReciprocalSetup energy across box scales.
// Prints the full reciprocal energy for a range of volume scale factors so
// we can confirm the energy landscape drives compression toward ~998 kg/m³.
// The scale factor applies uniformly to all box dimensions and all atom coords.
// At each scale the potential mesh is fully recomputed (same path as
// VolumeTransfer). After each probe the PME state is restored via exgMolCache.
// ---------------------------------------------------------------------------
TEST_F(EwaldPMEMovesTest, ReciprocalEnergyVsVolume) {
  Simulation sim("in.conf");
  EwaldPME *pme = dynamic_cast<EwaldPME *>(sim.GetEwald());
  ASSERT_NE(pme, nullptr);

  uint box = 0;
  sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();
  double refRecip = sim.GetSystemEnergy().boxEnergy[box].recip;

  // Scale factors: 1.20 = low density (~555 kg/m³), 0.95 = high density (~1122
  // kg/m³) Volume scales as scale^3, so scale 1.02→~998 kg/m³ (from 957 kg/m³
  // start)
  std::vector<double> scales = {1.20, 1.15, 1.10, 1.05, 1.02,
                                1.00, 0.98, 0.97, 0.95};

  std::cout << "\n=== PME ReciprocalEnergyVsVolume scan ===\n";
  std::cout << "Scale     Volume(A^3)   Recip(K)\n";

  double firstRecip = 0.0;
  double lastRecip = 0.0;
  bool isFirst = true;

  for (double scale : scales) {
    BoxDimensions &curAxes = sim.GetBoxDim();
    BoxDimensions newAxes = curAxes;
    double L0 = curAxes.axis.Get(box).x;
    double Lnew = L0 * scale;

    newAxes.axis.Set(box, XYZ(Lnew, Lnew, Lnew));
    newAxes.halfAx.Set(box, XYZ(Lnew / 2, Lnew / 2, Lnew / 2));
    newAxes.volume[box] = Lnew * Lnew * Lnew;
    newAxes.volInv[box] = 1.0 / newAxes.volume[box];
    newAxes.cellBasis[box].Set(0, XYZ(Lnew, 0.0, 0.0));
    newAxes.cellBasis[box].Set(1, XYZ(0.0, Lnew, 0.0));
    newAxes.cellBasis[box].Set(2, XYZ(0.0, 0.0, Lnew));

    XYZArray newCoords(sim.GetCoordinates().Count());
    for (uint i = 0; i < newCoords.Count(); ++i) {
      XYZ pos = sim.GetCoordinates().Get(i);
      newCoords.Set(i, XYZ(pos.x * scale, pos.y * scale, pos.z * scale));
    }

    pme->RecipInit(box, newAxes);
    pme->BoxReciprocalSetup(box, newCoords);
    double trialRecip = pme->BoxReciprocal(box, true);

    std::cout << std::fixed << std::setprecision(4) << scale << "      "
              << std::setprecision(1) << newAxes.volume[box] << "    "
              << trialRecip << "\n";

    if (isFirst) {
      firstRecip = trialRecip;
      isFirst = false;
    }
    lastRecip = trialRecip;

    // Restore PME state (simulate rejection)
    pme->exgMolCache();
  }

  std::cout << "Reference recip (scale=1.00, SystemTotal): " << refRecip
            << "\n";
  std::cout << "===========================================\n";

  // The energy should vary meaningfully across the scan (not degenerate/flat).
  // A flat landscape means PME is not sensing the volume change at all.
  double variation = std::abs(firstRecip - lastRecip);
  EXPECT_GT(variation, 1000.0)
      << "PME recip energy barely changes over 2.2x volume range — "
      << "possible normalization bug. variation=" << variation;
}

// ---------------------------------------------------------------------------
// Regression test for the K_trial/greenFunc_trial corruption bug.
//
// This test catches the specific failure mode where SumMeshEnergy uses
// K_trial and greenFunc_trial (which become stale after a rejected volume
// move) instead of the committed K and greenFunc arrays during per-move
// energy evaluations.
//
// The test sequence is:
//   1. Record the initial reciprocal energy via full rebuild.
//   2. Perform a volume move that changes K_trial to *different* dimensions.
//   3. Reject the volume move (via exgMolCache).
//   4. Perform N sequential displacement moves, accepting each one.
//   5. After each accepted move, verify the PME incremental energy matches
//      a full system rebuild within tight tolerance.
//
// Without the useTrial fix, step 5 would show growing discrepancies because
// ComputeDeltaSsq→SumMeshEnergy would use stale K_trial dimensions and the
// wrong Green function values.
// ---------------------------------------------------------------------------
TEST_F(EwaldPMEMovesTest, RejectedVolumeDoesNotCorruptPerMoveEnergy) {
  Simulation sim("in.conf");
  EwaldPME *pme = dynamic_cast<EwaldPME *>(sim.GetEwald());
  ASSERT_NE(pme, nullptr);

  uint box = 0;
  sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();
  double baselineRecip = sim.GetSystemEnergy().boxEnergy[box].recip;
  std::cout << "Baseline reciprocal energy: " << baselineRecip << " K\n";

  // --- Step 1: Perform a volume move that will change K_trial ---
  // Use a large scale (1.30) to force K_trial to differ from K.
  // With PMEGridSpacing=1.5: K = round(25.0/1.5) = 17;
  // after scale 1.30: K_trial = round(32.5/1.5) = 22. This is different.
  {
    BoxDimensions newAxes = sim.GetBoxDim();
    double scale = 1.30;
    double Lnew = newAxes.axis.Get(box).x * scale;
    newAxes.axis.Set(box, XYZ(Lnew, Lnew, Lnew));
    newAxes.halfAx.Set(box, XYZ(Lnew / 2, Lnew / 2, Lnew / 2));
    newAxes.volume[box] = Lnew * Lnew * Lnew;
    newAxes.volInv[box] = 1.0 / newAxes.volume[box];
    newAxes.cellBasis[box].Set(0, XYZ(Lnew, 0.0, 0.0));
    newAxes.cellBasis[box].Set(1, XYZ(0.0, Lnew, 0.0));
    newAxes.cellBasis[box].Set(2, XYZ(0.0, 0.0, Lnew));

    XYZArray scaledCoords(sim.GetCoordinates().Count());
    for (uint i = 0; i < scaledCoords.Count(); ++i) {
      XYZ pos = sim.GetCoordinates().Get(i);
      scaledCoords.Set(i, XYZ(pos.x * scale, pos.y * scale, pos.z * scale));
    }

    pme->RecipInit(box, newAxes);
    pme->BoxReciprocalSetup(box, scaledCoords);
    double trialEnergy = pme->BoxReciprocal(box, true);
    std::cout << "Trial volume energy (scale=1.30): " << trialEnergy << " K\n";

    // REJECT the volume move
    pme->exgMolCache();
    std::cout << "Volume move rejected. K_trial may be dirty.\n";
  }

  // --- Step 2: Perform N displacement moves and verify consistency ---
  const int nMoves = 5;
  // Use different molecules and displacement vectors for each move
  uint molIndices[] = {5, 10, 15, 20, 25};
  XYZ displacements[] = {XYZ(1.5, -0.8, 2.1), XYZ(-2.3, 1.4, -0.5),
                         XYZ(0.7, 2.5, -1.8), XYZ(-1.1, -1.1, 3.0),
                         XYZ(3.2, 0.3, -2.4)};

  for (int moveIdx = 0; moveIdx < nMoves; ++moveIdx) {
    uint molIndex = molIndices[moveIdx];
    if (molIndex >= sim.GetMolecules().count)
      molIndex = moveIdx; // Fallback to low index

    XYZ move = displacements[moveIdx];
    uint nAtoms = sim.GetMolecules().GetKind(molIndex).NumAtoms();
    XYZArray newCoords(nAtoms);
    uint startAtom = sim.GetMolecules().MolStart(molIndex);
    for (uint i = 0; i < nAtoms; ++i) {
      newCoords.Set(i, sim.GetCoordinates().Get(startAtom + i) + move);
    }

    // Calculate incremental dE
    double dE = pme->MolReciprocal(newCoords, molIndex, box);

    // Accept the move
    pme->UpdateRecip(box);
    double incrementalEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;

    // Update coordinates so full rebuild sees the new positions
    for (uint i = 0; i < nAtoms; ++i) {
      sim.GetCoordinates().Set(startAtom + i, newCoords.Get(i));
    }

    // Full rebuild to get ground-truth energy
    pme->UpdateVectorsAndRecipTerms(false);
    sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();
    double fullRebuildEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;

    std::cout << "Move " << moveIdx << " (mol " << molIndex << "): "
              << "incremental=" << incrementalEnergy
              << "  fullRebuild=" << fullRebuildEnergy
              << "  diff=" << (incrementalEnergy - fullRebuildEnergy) << "\n";

    // The incremental energy MUST match the full rebuild.
    // Without the useTrial fix, this would diverge badly after the
    // rejected volume move because SumMeshEnergy would read stale
    // K_trial dimensions and greenFunc_trial values.
    EXPECT_NEAR(incrementalEnergy, fullRebuildEnergy, 1e-1)
        << "Reciprocal energy corruption detected after rejected volume move! "
        << "Move " << moveIdx << ", mol " << molIndex;
  }
}

// ---------------------------------------------------------------------------
// Regression test: multiple rejected volume moves followed by displacement.
//
// This test is a stress variant that performs multiple rejected volume moves
// at different scales before running displacement moves. It catches bugs
// where internal state accumulates corruption across multiple rejections.
// ---------------------------------------------------------------------------
TEST_F(EwaldPMEMovesTest, MultipleRejectedVolumesDoNotCorruptEnergy) {
  Simulation sim("in.conf");
  EwaldPME *pme = dynamic_cast<EwaldPME *>(sim.GetEwald());
  ASSERT_NE(pme, nullptr);

  uint box = 0;
  sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();

  // Perform 3 rejected volume moves at different scales
  double scales[] = {1.20, 0.85, 1.40};
  for (double scale : scales) {
    BoxDimensions newAxes = sim.GetBoxDim();
    double Lnew = newAxes.axis.Get(box).x * scale;
    newAxes.axis.Set(box, XYZ(Lnew, Lnew, Lnew));
    newAxes.halfAx.Set(box, XYZ(Lnew / 2, Lnew / 2, Lnew / 2));
    newAxes.volume[box] = Lnew * Lnew * Lnew;
    newAxes.volInv[box] = 1.0 / newAxes.volume[box];
    newAxes.cellBasis[box].Set(0, XYZ(Lnew, 0.0, 0.0));
    newAxes.cellBasis[box].Set(1, XYZ(0.0, Lnew, 0.0));
    newAxes.cellBasis[box].Set(2, XYZ(0.0, 0.0, Lnew));

    XYZArray scaledCoords(sim.GetCoordinates().Count());
    for (uint i = 0; i < scaledCoords.Count(); ++i) {
      XYZ pos = sim.GetCoordinates().Get(i);
      scaledCoords.Set(i, XYZ(pos.x * scale, pos.y * scale, pos.z * scale));
    }

    pme->RecipInit(box, newAxes);
    pme->BoxReciprocalSetup(box, scaledCoords);
    pme->BoxReciprocal(box, true);
    pme->exgMolCache(); // REJECT
    std::cout << "Rejected volume move at scale=" << scale << "\n";
  }

  // Now do a displacement move and verify
  uint molIndex = 8;
  if (molIndex >= sim.GetMolecules().count)
    molIndex = 0;

  uint nAtoms = sim.GetMolecules().GetKind(molIndex).NumAtoms();
  XYZArray newCoords(nAtoms);
  uint startAtom = sim.GetMolecules().MolStart(molIndex);
  XYZ move(2.0, -1.5, 0.8);
  for (uint i = 0; i < nAtoms; ++i) {
    newCoords.Set(i, sim.GetCoordinates().Get(startAtom + i) + move);
  }

  double dE = pme->MolReciprocal(newCoords, molIndex, box);
  pme->UpdateRecip(box);
  double incrementalEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;

  for (uint i = 0; i < nAtoms; ++i) {
    sim.GetCoordinates().Set(startAtom + i, newCoords.Get(i));
  }

  pme->UpdateVectorsAndRecipTerms(false);
  sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();
  double fullRebuildEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;

  std::cout << "After 3 rejected volumes + 1 displacement: "
            << "incremental=" << incrementalEnergy
            << "  fullRebuild=" << fullRebuildEnergy
            << "  diff=" << (incrementalEnergy - fullRebuildEnergy) << "\n";

  EXPECT_NEAR(incrementalEnergy, fullRebuildEnergy, 1e-1)
      << "Reciprocal energy corruption after multiple rejected volume moves!";
}

TEST_F(EwaldPMEMovesTest, RejectedDisplacementConsistency) {
  Simulation sim("in.conf");
  EwaldPME *pme = dynamic_cast<EwaldPME *>(sim.GetEwald());
  ASSERT_NE(pme, nullptr);

  uint box = 0;
  sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();
  double baselineEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;

  uint molIndex = 11;
  if (molIndex >= sim.GetMolecules().count)
    molIndex = 0;

  XYZ move(1.2, -1.8, 0.4);
  uint nAtoms = sim.GetMolecules().GetKind(molIndex).NumAtoms();
  XYZArray newCoords(nAtoms);
  uint startAtom = sim.GetMolecules().MolStart(molIndex);
  for (uint i = 0; i < nAtoms; ++i) {
    newCoords.Set(i, sim.GetCoordinates().Get(startAtom + i) + move);
  }

  // Evaluate incremental dE but REJECT it
  double dE = pme->MolReciprocal(newCoords, molIndex, box);
  pme->exgMolCache();

  // Full rebuild to assert the reject left the original system perfectly intact
  pme->UpdateVectorsAndRecipTerms(false);
  sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();
  double fullRebuildEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;

  EXPECT_NEAR(baselineEnergy, fullRebuildEnergy, 1e-1)
      << "Reciprocal energy changed after a rejected displacement move! "
         "MolReciprocal mutated core arrays incorrectly.";
}

TEST_F(EwaldPMEMovesTest, SmallVolumeMoveConsistency) {
  Simulation sim("in.conf");
  EwaldPME *pme = dynamic_cast<EwaldPME *>(sim.GetEwald());
  ASSERT_NE(pme, nullptr);

  uint box = 0;
  sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();
  double initialEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;

  // Use a tiny 1.0001 volume scaling factor that will NOT trigger kChanged =
  // true prior to the RecipInit lock. This explicitly tests the stale
  // greenFunc_trial bug.
  BoxDimensions newAxes = sim.GetBoxDim();
  double scale = 1.0001;
  double Lnew = newAxes.axis.Get(box).x * scale;
  newAxes.axis.Set(box, XYZ(Lnew, Lnew, Lnew));
  newAxes.halfAx.Set(box, XYZ(Lnew / 2, Lnew / 2, Lnew / 2));
  newAxes.volume[box] = Lnew * Lnew * Lnew;
  newAxes.volInv[box] = 1.0 / newAxes.volume[box];
  newAxes.cellBasis[box].Set(0, XYZ(Lnew, 0.0, 0.0));
  newAxes.cellBasis[box].Set(1, XYZ(0.0, Lnew, 0.0));
  newAxes.cellBasis[box].Set(2, XYZ(0.0, 0.0, Lnew));

  XYZArray newCoords(sim.GetCoordinates().Count());
  for (uint i = 0; i < newCoords.Count(); ++i) {
    XYZ pos = sim.GetCoordinates().Get(i);
    newCoords.Set(i, XYZ(pos.x * scale, pos.y * scale, pos.z * scale));
  }

  pme->RecipInit(box, newAxes);
  pme->BoxReciprocalSetup(box, newCoords);
  double trialEnergy = pme->BoxReciprocal(box, true);

  // Accept the move
  for (uint i = 0; i < newCoords.Count(); ++i) {
    sim.GetCoordinates().Set(i, newCoords.Get(i));
  }
  sim.GetBoxDim() = newAxes;
  sim.GetSystemEnergy().boxEnergy[box].recip = trialEnergy;
  pme->UpdateRecip(box);
  pme->UpdateRecipVec(box);

  pme->UpdateVectorsAndRecipTerms(false);
  sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();
  double fullRebuildEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;

  EXPECT_NEAR(trialEnergy, fullRebuildEnergy, 1e-1)
      << "Reciprocal energy from small volumetric scaled trial does not match "
         "full rebuild due to stale block bugs.";
}

TEST_F(EwaldPMEMovesTest, DynamicGridResizingRejectionConsistency) {
  uint box = 0;
  Simulation sim("in.conf");
  EwaldPME *pme = static_cast<EwaldPME *>(sim.GetEwald());

  pme->UpdateVectorsAndRecipTerms(false);
  sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();

  // Pick a molecule and calculate a tiny displacement to get a baseline
  // DeltaERecip
  uint molIndex = 0;
  XYZArray newCoords(sim.GetCoordinates().Count());
  for (uint i = 0; i < newCoords.Count(); ++i) {
    if (i >= sim.GetMolecules().MolStart(molIndex) &&
        i < sim.GetMolecules().MolStart(molIndex) +
                sim.GetMolecules().MolLength(molIndex)) {
      XYZ pos = sim.GetCoordinates().Get(i);
      newCoords.Set(i, XYZ(pos.x + 0.1, pos.y + 0.1, pos.z + 0.1));
    } else {
      newCoords.Set(i, sim.GetCoordinates().Get(i));
    }
  }

  double expected_dE = pme->MolReciprocal(newCoords, molIndex, box);
  pme->exgMolCache(); // reject displacement

  // Now perform a volume scaling that guarantees a K lattice dimension shift
  BoxDimensions newAxes = sim.GetBoxDim();
  double scale =
      1.08; // Expand by 8% to cross K boundaries without exceeding Kmax buffer
  double Lnew = newAxes.axis.Get(box).x * scale;
  newAxes.axis.Set(box, XYZ(Lnew, Lnew, Lnew));
  newAxes.halfAx.Set(box, XYZ(Lnew / 2, Lnew / 2, Lnew / 2));
  newAxes.volume[box] = Lnew * Lnew * Lnew;
  newAxes.volInv[box] = 1.0 / newAxes.volume[box];
  newAxes.cellBasis[box].Set(0, XYZ(Lnew, 0.0, 0.0));
  newAxes.cellBasis[box].Set(1, XYZ(0.0, Lnew, 0.0));
  newAxes.cellBasis[box].Set(2, XYZ(0.0, 0.0, Lnew));

  XYZArray expandedCoords(sim.GetCoordinates().Count());
  for (uint i = 0; i < expandedCoords.Count(); ++i) {
    XYZ pos = sim.GetCoordinates().Get(i);
    expandedCoords.Set(i, XYZ(pos.x * scale, pos.y * scale, pos.z * scale));
  }

  // Trigger the destructive K trial resizing!
  pme->RecipInit(box, newAxes);
  pme->BoxReciprocalSetup(box, expandedCoords);
  pme->BoxReciprocal(box, true);

  // REJECT the Volume move.
  // If `K_allocated` tracking and forceful grid restitution works, the FFT
  // arrays are restored to the committed boundaries.
  pme->exgMolCache();

  // Test the arrays by executing the exact same displacement move!
  // If the committed fwdPlan or S_ref were silently corrupted due to the
  // bounding resizing, we get garbage or segfault.
  double recovered_dE = pme->MolReciprocal(newCoords, molIndex, box);
  pme->exgMolCache();

  EXPECT_NEAR(expected_dE, recovered_dE, 1e-4)
      << "Committed K grid was strictly corrupted and not successfully "
         "reconstructed after a rejected dynamic-bounding volume move.";
}

TEST_F(EwaldPMEMovesTest, ConstantGridResizingRejectionConsistency) {
  uint box = 0;
  Simulation sim("in.conf");
  EwaldPME *pme = static_cast<EwaldPME *>(sim.GetEwald());

  pme->UpdateVectorsAndRecipTerms(false);
  sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();

  // Pick a molecule and calculate a tiny displacement to get a baseline
  // DeltaERecip
  uint molIndex = 0;
  XYZArray newCoords(sim.GetCoordinates().Count());
  for (uint i = 0; i < newCoords.Count(); ++i) {
    if (i >= sim.GetMolecules().MolStart(molIndex) &&
        i < sim.GetMolecules().MolStart(molIndex) +
                sim.GetMolecules().MolLength(molIndex)) {
      XYZ pos = sim.GetCoordinates().Get(i);
      newCoords.Set(i, XYZ(pos.x + 0.1, pos.y + 0.1, pos.z + 0.1));
    } else {
      newCoords.Set(i, sim.GetCoordinates().Get(i));
    }
  }

  double expected_dE = pme->MolReciprocal(newCoords, molIndex, box);
  pme->exgMolCache(); // reject displacement

  // Now perform a microscopic volume scaling that guarantees the K lattice
  // dimension DOES NOT shift
  BoxDimensions newAxes = sim.GetBoxDim();
  double scale =
      1.0001; // Tiny expansion strictly constrained to identical K bin limits
  double Lnew = newAxes.axis.Get(box).x * scale;
  newAxes.axis.Set(box, XYZ(Lnew, Lnew, Lnew));
  newAxes.halfAx.Set(box, XYZ(Lnew / 2, Lnew / 2, Lnew / 2));
  newAxes.volume[box] = Lnew * Lnew * Lnew;
  newAxes.volInv[box] = 1.0 / newAxes.volume[box];
  newAxes.cellBasis[box].Set(0, XYZ(Lnew, 0.0, 0.0));
  newAxes.cellBasis[box].Set(1, XYZ(0.0, Lnew, 0.0));
  newAxes.cellBasis[box].Set(2, XYZ(0.0, 0.0, Lnew));

  XYZArray expandedCoords(sim.GetCoordinates().Count());
  for (uint i = 0; i < expandedCoords.Count(); ++i) {
    XYZ pos = sim.GetCoordinates().Get(i);
    expandedCoords.Set(i, XYZ(pos.x * scale, pos.y * scale, pos.z * scale));
  }

  // Trigger the trial setup that populates the chargeMesh with the modified
  // scale structure
  pme->RecipInit(box, newAxes);
  pme->BoxReciprocalSetup(box, expandedCoords);
  pme->BoxReciprocal(box, true);

  // REJECT the Volume move.
  // We recently enforced that even if `K_allocated` remains identical,
  // exgMolCache explicitly checks volume divergence to reconstruct the smeared
  // mesh limits cleanly.
  pme->exgMolCache();

  // Re-run the baseline test: If chargeMesh reconstruction failed, DeltaERecip
  // uses the ghost expanded matrix coefficients to evaluate the differential
  // displacement and wildly fails.
  double recovered_dE = pme->MolReciprocal(newCoords, molIndex, box);
  pme->exgMolCache();

  EXPECT_NEAR(expected_dE, recovered_dE, 1e-4)
      << "ChargeMesh retained Ghost Atoms: Mathematical Fourier reconstruction "
         "explicitly bypassed after constant-K trial regression.";
}
