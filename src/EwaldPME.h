#pragma once
#include "Ewald.h"
#include <fftw3.h>
#include <vector>

class EwaldPME : public Ewald {
public:
  EwaldPME(StaticVals &stat, System &sys);
  virtual ~EwaldPME();

  // Full-box path (FFT)
  void Init() override;
  void AllocMem() override;
  void BoxReciprocalSetup(uint box, XYZArray const &molCoords) override;
  void BoxReciprocalSums(uint box, XYZArray const &molCoords) override;
  double BoxReciprocal(uint box, bool isNewVolume) const override;
  void UpdateRecipVec(uint box) override; // recomputes C(m)
  void SetRecipRef(uint box) override { UpdateRecipVec(box); }
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
  fftw_plan fwdPlan[BOX_TOTAL];
  uint refreshInterval; // steps between full S_ref recomputes

  // NPT / Volume move trial state
  fftw_complex **S_trial;
  double **greenFunc_trial;
  int K_trial[BOX_TOTAL][3];
  double tempEnergyRecip[BOX_TOTAL];
  Virial tempVirialRecip[BOX_TOTAL];

  void ComputeChargeMesh(uint box, XYZArray const &molCoords);
  void AddMoleculeToMesh(uint box, uint molIndex, XYZArray const &molCoords,
                         double *mesh);
  void ComputeGreenFunction(uint box);
  double SumMeshEnergy(uint box, const fftw_complex *S,
                       Virial *virial = nullptr) const;

  // Accumulate ΔS(m) for a molecule's atoms into dS (half-complex array)
  void ComputeMolDeltaS(uint box, const XYZArray &coords, const uint *atomIdx,
                        const double *charges, uint nAtoms, double sign,
                        fftw_complex *dS) const;

  void AddAtomToDeltaS(uint box, uint atomIdx, double charge,
                       const XYZArray &coords, fftw_complex *dS) const;

  double DeltaERecip(uint box, const XYZArray *newCoords, double sign_new,
                     const XYZArray *oldCoords, double sign_old,
                     const uint *atomIndices, const double *charges,
                     uint nAtoms, bool updateSRef);
};
