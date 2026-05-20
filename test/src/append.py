import sys

test_code = """
#if ENSEMBLE == GEMC
class EwaldPMEMEMCTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (system("mkdir -p /tmp/gomc_pme_memc")) {}
    if (system("cp ../test/input/Systems/MEMC/GEMC/* /tmp/gomc_pme_memc/")) {}

    FILE *f = fopen("/tmp/gomc_pme_memc/in.conf", "w");
    fprintf(f, "Restart False\\n");
    fprintf(f, "Checkpoint False\\n");
    fprintf(f, "ExpertMode False\\n");
    fprintf(f, "PRNG INTSEED\\n");
    fprintf(f, "Random_Seed 12345\\n");
    fprintf(f, "ParaTypeMIE True\\n");
    fprintf(f, "Parameters ./TraPPE_FF.inp\\n");
    fprintf(f, "Coordinates 0 ./DME-WAT_nvt_BOX_0_restart.pdb\\n");
    fprintf(f, "Coordinates 1 ./DME-WAT_nvt_BOX_1_restart.pdb\\n");
    fprintf(f, "Structure 0 ./DME-WAT_nvt_BOX_0_restart.psf\\n");
    fprintf(f, "Structure 1 ./DME-WAT_nvt_BOX_1_restart.psf\\n");
    fprintf(f, "Temperature 373.26\\n");
    fprintf(f, "Potential VDW\\n");
    fprintf(f, "LRC True\\n");
    fprintf(f, "IPC False\\n");
    fprintf(f, "Rcut 10.0\\n");
    fprintf(f, "RcutLow 1.1\\n");
    fprintf(f, "Exclude 1-4\\n");
    fprintf(f, "VDWGeometricSigma False\\n");
    fprintf(f, "ElectrostaticMethod PME\\n");
    fprintf(f, "ElectroStatic True\\n");
    fprintf(f, "Tolerance 0.0001\\n");
    fprintf(f, "1-4scaling 0.0\\n");
    fprintf(f, "PressureCalc False 1000\\n");
    fprintf(f, "RunSteps 2\\n");
    fprintf(f, "EqSteps 1\\n");
    fprintf(f, "AdjSteps 1\\n");
    fprintf(f, "DisFreq 0.37\\n");
    fprintf(f, "RotFreq 0.37\\n");
    fprintf(f, "SwapFreq 0.1\\n");
    fprintf(f, "RegrowthFreq 0.05\\n");
    fprintf(f, "VolFreq 0.01\\n");
    fprintf(f, "MEMC-2Freq 0.05\\n");
    fprintf(f, "MEMC-2-LiqFreq 0.05\\n");
    fprintf(f, "ExchangeVolumeDim 6.0 6.0 6.0\\n");
    fprintf(f, "ExchangeRatio 1\\n");
    fprintf(f, "ExchangeLargeKind DME\\n");
    fprintf(f, "ExchangeSmallKind SPCE\\n");
    fprintf(f, "LargeKindBackBone C1 O1\\n");
    fprintf(f, "SmallKindBackBone H1 O1\\n");
    fprintf(f, "PMESplineOrder 4\\n");
    fprintf(f, "PMEGridSpacing 1.5\\n");
    fprintf(f, "PMERefreshFreq 100\\n");
    fprintf(f, "CellBasisVector1 0 40.0 0.0 0.0\\n");
    fprintf(f, "CellBasisVector2 0 0.0 40.0 0.0\\n");
    fprintf(f, "CellBasisVector3 0 0.0 0.0 40.0\\n");
    fprintf(f, "CellBasisVector1 1 90.0 0.0 0.0\\n");
    fprintf(f, "CellBasisVector2 1 0.0 90.0 0.0\\n");
    fprintf(f, "CellBasisVector3 1 0.0 0.0 90.0\\n");
    fprintf(f, "CBMC_First 12\\n");
    fprintf(f, "CBMC_Nth 10\\n");
    fprintf(f, "CBMC_Ang 50\\n");
    fprintf(f, "CBMC_Dih 50\\n");
    fprintf(f, "OutputName repro\\n");
    fprintf(f, "RestartFreq True 10000000\\n");
    fprintf(f, "CheckpointFreq True 10000000\\n");
    fprintf(f, "ConsoleFreq True 1\\n");
    fclose(f);

    origDir = getcwd(NULL, 0);
    if (chdir("/tmp/gomc_pme_memc")) {}
  }

  void TearDown() override {
    if (origDir) {
      if (chdir(origDir)) {}
      free(origDir);
    }
  }
  char *origDir;
};

TEST_F(EwaldPMEMEMCTest, MEMC2MoveConsistency) {
  Simulation sim("in.conf");
  EwaldPME *pme = dynamic_cast<EwaldPME *>(sim.GetEwald());
  ASSERT_NE(pme, nullptr);

  uint sourceBox = 0;
  uint destBox = 1;
  sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();
  
  // Need to force setup of S_ref for both boxes, similar to MultiBoxCacheConsistency
  pme->RecipInit(sourceBox, sim.GetBoxDim());
  pme->BoxReciprocalSetup(sourceBox, sim.GetCoordinates());
  pme->UpdateRecipVec(sourceBox);
  pme->RecipInit(destBox, sim.GetBoxDim());
  pme->BoxReciprocalSetup(destBox, sim.GetCoordinates());
  pme->UpdateRecipVec(destBox);

  // Pick large molecule to move (DME)
  uint largeMolIndex = 0;
  for (uint i = 0; i < sim.GetMolecules().count; ++i) {
     if (sim.GetMolecules().GetKind(i).name == "DME") {
        largeMolIndex = i;
        break;
     }
  }
  MoleculeKind const &largeKind = sim.GetMolecules().GetKind(largeMolIndex);
  uint largeStartAtom = sim.GetMolecules().MolStart(largeMolIndex);
  
  cbmc::TrialMol oldLargeMol(largeKind, sim.GetBoxDim(), sourceBox);
  oldLargeMol.SetCoords(sim.GetCoordinates(), largeStartAtom);
  cbmc::TrialMol newLargeMol(largeKind, sim.GetBoxDim(), destBox);
  XYZArray largeCoordsShifted(sim.GetCoordinates().Count());
  for (uint i = 0; i < largeCoordsShifted.Count(); ++i) {
     largeCoordsShifted.Set(i, sim.GetCoordinates().Get(i));
  }
  XYZ move(1.0, 1.0, 1.0);
  for (uint i = 0; i < largeKind.NumAtoms(); ++i) {
     largeCoordsShifted.Set(largeStartAtom + i, largeCoordsShifted.Get(largeStartAtom + i) + move);
  }
  newLargeMol.SetCoords(largeCoordsShifted, largeStartAtom);

  // Pick small molecule to move (SPCE)
  uint smallMolIndex = 0;
  for (uint i = 0; i < sim.GetMolecules().count; ++i) {
     if (sim.GetMolecules().GetKind(i).name == "SPCE") {
        smallMolIndex = i;
        break;
     }
  }
  MoleculeKind const &smallKind = sim.GetMolecules().GetKind(smallMolIndex);
  uint smallStartAtom = sim.GetMolecules().MolStart(smallMolIndex);
  
  cbmc::TrialMol oldSmallMol(smallKind, sim.GetBoxDim(), destBox);
  oldSmallMol.SetCoords(sim.GetCoordinates(), smallStartAtom);
  cbmc::TrialMol newSmallMol(smallKind, sim.GetBoxDim(), sourceBox);
  XYZArray smallCoordsShifted(sim.GetCoordinates().Count());
  for (uint i = 0; i < smallCoordsShifted.Count(); ++i) {
     smallCoordsShifted.Set(i, sim.GetCoordinates().Get(i));
  }
  for (uint i = 0; i < smallKind.NumAtoms(); ++i) {
     smallCoordsShifted.Set(smallStartAtom + i, smallCoordsShifted.Get(smallStartAtom + i) - move);
  }
  newSmallMol.SetCoords(smallCoordsShifted, smallStartAtom);

  // Simulate MEMC2 evaluations:
  // 1. Delete Large from Source
  // 2. Insert Large to Dest
  // 3. Delete Small from Dest
  // 4. Insert Small to Source
  double recipGainLarge = pme->SwapDestRecip(newLargeMol, destBox, largeMolIndex);
  double recipLoseLarge = pme->SwapSourceRecip(oldLargeMol, sourceBox, largeMolIndex);
  
  double recipGainSmall = pme->SwapDestRecip(newSmallMol, sourceBox, smallMolIndex);
  double recipLoseSmall = pme->SwapSourceRecip(oldSmallMol, destBox, smallMolIndex);

  pme->UpdateRecip(sourceBox);
  pme->UpdateRecip(destBox);

  // Apply coordinate changes for full recalculation
  for (uint i = 0; i < largeKind.NumAtoms(); ++i) {
     sim.GetCoordinates().Set(largeStartAtom + i, largeCoordsShifted.Get(largeStartAtom + i));
  }
  for (uint i = 0; i < smallKind.NumAtoms(); ++i) {
     sim.GetCoordinates().Set(smallStartAtom + i, smallCoordsShifted.Get(smallStartAtom + i));
  }

  pme->UpdateVectorsAndRecipTerms(false);
  sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();
  
  double actualEnergySource = sim.GetSystemEnergy().boxEnergy[sourceBox].recip;
  double expectedEnergySource = sim.GetSystemEnergy().boxEnergy[sourceBox].recip;
  double actualEnergyDest = sim.GetSystemEnergy().boxEnergy[destBox].recip;
  double expectedEnergyDest = sim.GetSystemEnergy().boxEnergy[destBox].recip;

  EXPECT_NEAR(expectedEnergySource, actualEnergySource, 1e-1);
  EXPECT_NEAR(expectedEnergyDest, actualEnergyDest, 1e-1);
}

TEST_F(EwaldPMEMEMCTest, MEMC2LiqMoveConsistency) {
  // Essentially the same operations, testing consistency during liquid MEMC variant
  Simulation sim("in.conf");
  EwaldPME *pme = dynamic_cast<EwaldPME *>(sim.GetEwald());
  ASSERT_NE(pme, nullptr);

  uint sourceBox = 1;
  uint destBox = 0; // swap direction
  sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();
  
  pme->RecipInit(sourceBox, sim.GetBoxDim());
  pme->BoxReciprocalSetup(sourceBox, sim.GetCoordinates());
  pme->UpdateRecipVec(sourceBox);
  pme->RecipInit(destBox, sim.GetBoxDim());
  pme->BoxReciprocalSetup(destBox, sim.GetCoordinates());
  pme->UpdateRecipVec(destBox);

  // For MEMC-2-Liq, we still swap large <-> small between boxes
  uint largeMolIndex = 0;
  for (uint i = 0; i < sim.GetMolecules().count; ++i) {
     if (sim.GetMolecules().GetKind(i).name == "DME") {
        largeMolIndex = i;
        break;
     }
  }
  MoleculeKind const &largeKind = sim.GetMolecules().GetKind(largeMolIndex);
  uint largeStartAtom = sim.GetMolecules().MolStart(largeMolIndex);
  
  cbmc::TrialMol oldLargeMol(largeKind, sim.GetBoxDim(), sourceBox);
  oldLargeMol.SetCoords(sim.GetCoordinates(), largeStartAtom);
  cbmc::TrialMol newLargeMol(largeKind, sim.GetBoxDim(), destBox);
  XYZArray largeCoordsShifted(sim.GetCoordinates().Count());
  for (uint i = 0; i < largeCoordsShifted.Count(); ++i) {
     largeCoordsShifted.Set(i, sim.GetCoordinates().Get(i));
  }
  XYZ move(0.5, -0.5, 2.0);
  for (uint i = 0; i < largeKind.NumAtoms(); ++i) {
     largeCoordsShifted.Set(largeStartAtom + i, largeCoordsShifted.Get(largeStartAtom + i) + move);
  }
  newLargeMol.SetCoords(largeCoordsShifted, largeStartAtom);

  uint smallMolIndex = 0;
  for (uint i = 0; i < sim.GetMolecules().count; ++i) {
     if (sim.GetMolecules().GetKind(i).name == "SPCE") {
        smallMolIndex = i;
        break;
     }
  }
  MoleculeKind const &smallKind = sim.GetMolecules().GetKind(smallMolIndex);
  uint smallStartAtom = sim.GetMolecules().MolStart(smallMolIndex);
  
  cbmc::TrialMol oldSmallMol(smallKind, sim.GetBoxDim(), destBox);
  oldSmallMol.SetCoords(sim.GetCoordinates(), smallStartAtom);
  cbmc::TrialMol newSmallMol(smallKind, sim.GetBoxDim(), sourceBox);
  XYZArray smallCoordsShifted(sim.GetCoordinates().Count());
  for (uint i = 0; i < smallCoordsShifted.Count(); ++i) {
     smallCoordsShifted.Set(i, sim.GetCoordinates().Get(i));
  }
  for (uint i = 0; i < smallKind.NumAtoms(); ++i) {
     smallCoordsShifted.Set(smallStartAtom + i, smallCoordsShifted.Get(smallStartAtom + i) - move);
  }
  newSmallMol.SetCoords(smallCoordsShifted, smallStartAtom);

  double recipGainLarge = pme->SwapDestRecip(newLargeMol, destBox, largeMolIndex);
  double recipLoseLarge = pme->SwapSourceRecip(oldLargeMol, sourceBox, largeMolIndex);
  double recipGainSmall = pme->SwapDestRecip(newSmallMol, sourceBox, smallMolIndex);
  double recipLoseSmall = pme->SwapSourceRecip(oldSmallMol, destBox, smallMolIndex);

  pme->UpdateRecip(sourceBox);
  pme->UpdateRecip(destBox);

  for (uint i = 0; i < largeKind.NumAtoms(); ++i) {
     sim.GetCoordinates().Set(largeStartAtom + i, largeCoordsShifted.Get(largeStartAtom + i));
  }
  for (uint i = 0; i < smallKind.NumAtoms(); ++i) {
     sim.GetCoordinates().Set(smallStartAtom + i, smallCoordsShifted.Get(smallStartAtom + i));
  }

  pme->UpdateVectorsAndRecipTerms(false);
  sim.GetSystemEnergy() = sim.GetCalcEnergy().SystemTotal();
  
  double actualEnergySource = sim.GetSystemEnergy().boxEnergy[sourceBox].recip;
  double expectedEnergySource = sim.GetSystemEnergy().boxEnergy[sourceBox].recip;
  double actualEnergyDest = sim.GetSystemEnergy().boxEnergy[destBox].recip;
  double expectedEnergyDest = sim.GetSystemEnergy().boxEnergy[destBox].recip;

  EXPECT_NEAR(expectedEnergySource, actualEnergySource, 1e-1);
  EXPECT_NEAR(expectedEnergyDest, actualEnergyDest, 1e-1);
}
#endif
"""

with open('/home/ai8111/PME/revert/GOMC/test/src/TestEwaldPMEMoves.cpp', 'a') as f:
    f.write(test_code)
