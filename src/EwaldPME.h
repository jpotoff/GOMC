#pragma once
#include "Ewald.h"
#include "BoxDimensions.h"
#include <fftw3.h>
#include <vector>
#include <cassert>

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
  void UpdateRecipVec(uint box) override; // recomputes C(m)
  void SetRecipRef(uint box) override { UpdateRecipVec(box); }
  void UpdateRecip(uint box) override; // applies accepted moves to mesh
  Virial VirialReciprocal(Virial &virial, uint box) const override;

  // Per-move path (no FFT, O(P³))
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
  int pmeOrder;         // B-spline order P
  int K[BOX_TOTAL][3];  // mesh dims per box
  fftw_complex **S_ref; // stored FFT(Q), one per box (flat array)
  double **greenFunc;   // C(m), one per box
  double **chargeMesh;  // real staging buffer for FFT
  double **potentialMesh; // real mesh for background potential (after IFFT)
  fftw_plan fwdPlan[BOX_TOTAL];
  fftw_plan bwdPlan[BOX_TOTAL];
  uint refreshInterval; // steps between full S_ref recomputes

  // Scratch buffers for incremental changes
  double **scratchMesh;
  fftw_complex **S_delta;
  fftw_plan scratchPlan[BOX_TOTAL];

  // NPT / Volume move trial state
  BoxDimensions trialAxes[BOX_TOTAL];
  fftw_complex **S_trial;
  double **greenFunc_trial;
  int K_trial[BOX_TOTAL][3];
  double tempEnergyRecip[BOX_TOTAL];
  Virial tempVirialRecip[BOX_TOTAL];

  // Move caching for O(P³) incremental S_ref updates on accept
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

  void ComputeChargeMesh(uint box, XYZArray const &molCoords);
  void AddMoleculeToMesh(uint box, uint molIndex, XYZArray const &molCoords,
                         double *mesh);
  void ComputeGreenFunction(uint box);
  double SumMeshEnergy(uint box, const fftw_complex *S,
                       Virial *virial = nullptr) const;
  void UpdatePotentialMesh(uint box);
  double InterpolatePotential(uint box, const XYZ &coords) const;

  // Compute 0.5 * sum_m G(m) * |dS(m)|^2 where dS is the charge mesh delta
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
