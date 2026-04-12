#pragma once
#include <cassert>
#include <fftw3.h>
#include <vector>

#include "BasicTypes.h"
#include "Ewald.h"
#include "XYZArray.h"
#include "EnergyTypes.h"

class EwaldPME : public Ewald {
public:
  EwaldPME(StaticVals &stat, System &sys);
  virtual ~EwaldPME();

  // Full-box path (FFT)
  void Init() override;
  void AllocMem() override;
  void RecipInit(uint box, BoxDimensions const &axes) override;
  void BoxReciprocalSetup(uint box, XYZArray const &molCoords) override;
  void BoxReciprocalSums(uint box, XYZArray const &molCoords) override;
  double BoxReciprocal(uint box, bool isNewVolume) const override;
  void UpdateRecipVec(uint box) override; 
  void SetRecipRef(uint box) override { UpdateRecipVec(box); }
  void UpdateRecip(uint box) override; 
  void Maintain(const ulong step) override;
  void exgMolCache() override; 
  void backupMolCache() override;
  void CopyRecip(uint box) override;
  void RestoreMol(int molIndex) override; 
  void BoxForceReciprocal(XYZArray const &molCoords, XYZArray &atomForceRec,
                          XYZArray &molForceRec, uint box) override;
  Virial VirialReciprocal(Virial &virial, uint box) const override;

  // Per-move path (no FFT)
  double MolReciprocal(XYZArray const &molCoords, const uint molIndex,
                       const uint box) override;
  double SwapDestRecip(const cbmc::TrialMol &newMol, const uint box,
                       const int molIndex) override;
  double SwapSourceRecip(const cbmc::TrialMol &oldMol, const uint box,
                         const int molIndex) override;
  double ChangeLambdaRecip(XYZArray const &molCoords, const double lambdaOld,
                           const double lambdaNew, const uint molIndex,
                           const uint box) override;
  double MolExchangeReciprocal(const std::vector<cbmc::TrialMol> &newMol,
                               const std::vector<cbmc::TrialMol> &oldMol,
                               const std::vector<uint> &molIndexNew,
                               const std::vector<uint> &molIndexOld,
                               bool first_call) override;

private:
  int pmeOrder;           
  int K[BOX_TOTAL][3];    
  fftw_complex **S_ref;   
  double **greenFunc;     
  double **chargeMesh;    
  double **potentialMesh; 
  fftw_plan fwdPlan[BOX_TOTAL];
  fftw_plan bwdPlan[BOX_TOTAL];
  uint refreshInterval; 

  double **scratchMesh;
  fftw_complex **S_delta;
  fftw_plan scratchPlan[BOX_TOTAL];

  BoxDimensions trialAxes[BOX_TOTAL]; 
  fftw_complex **S_trial;
  double **greenFunc_trial;
  int K_trial[BOX_TOTAL][3];
  double tempEnergyRecip[BOX_TOTAL];
  Virial tempVirialRecip[BOX_TOTAL];

  bool pendingUpdate;
  bool forceFullUpdate;
  uint cachedBox;
  XYZArray cachedNewCoords;
  XYZArray cachedOldCoords;
  double cachedSignNew;
  double cachedSignOld;
  std::vector<uint> cachedAtomIndices;
  std::vector<double> cachedCharges;
  uint cachedNAtoms;

  void UpdateGreenFunction(uint box, const BoxDimensions &axes, double *gf_out);
  double SumMeshEnergy(uint box, fftw_complex *S,
                       Virial *virial = nullptr,
                       bool useTrial = true) const;
  void UpdatePotentialMesh(uint box);
  double InterpolatePotential(uint box, const XYZ &coords) const;

  double ComputeDeltaSsq(uint box, const XYZArray *newCoords, double sign_new,
                         const XYZArray *oldCoords, double sign_old,
                         const uint *atomIndices, const double *charges,
                         uint nAtoms) const;

  void UpdateAtomInMesh(uint box, const double *charges, uint nAtoms,
                        const XYZArray &coords, double sign);

  double DeltaERecip(uint box, const XYZArray *newCoords, double sign_new,
                     const XYZArray *oldCoords, double sign_old,
                     const uint *atomIndices, const double *charges,
                     uint nAtoms, bool updateSRef);
};
