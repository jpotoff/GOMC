/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#ifndef FF_EXP6_H
#define FF_EXP6_H

#include "BasicTypes.h" //for uint
#include "FFConst.h"    //constants related to particles.
#include "FFAdapter.h"
#include "MiePotential.h"
#include "CoulEvaluators.h"
#include "VdwEvaluators.h"
#include "NumLib.h" //For Cb, Sq
#ifdef GOMC_CUDA
#include "ConstantDefinitionsCUDAKernel.cuh"
#endif

//////////////////////////////////////////////////////////////////////
//////////////////////////// Exp-6 Style /////////////////////////////
//////////////////////////////////////////////////////////////////////
// Virial and LJ potential calculation:
// U(rij)= expConst * ( (6/alpha) * exp( alpha * [1-(r/rmin)] )-(rmin/r)^6) )
// expConst=( eps-ij * alpha )/(alpha-6)
//
// Vir(r)= -du/dr * r
// Vir(r)= 6 * expConst * [(r/rmin) * exp(alpha * [1-(r/rmin)])-(rmin/r)^6]/ r^2
//

// `final` lets the compiler resolve this class's own virtual calls --
// notably the 4-argument CalcEn/CalcCoulomb calling their 2-argument
// counterparts -- statically, which is what allows the templated energy
// kernels to inline the pair math. Nothing derives from these.
struct FF_EXP6 final : public FFAdapter<FF_EXP6, ff::Exp6Eval, ff::PlainCoul> {
  friend struct FFTestAccess;

public:
  FF_EXP6(Forcefield &ff)
      : FFAdapter(ff), expConst(NULL), expConst_1_4(NULL), rMin(NULL),
        rMin_1_4(NULL), rMaxSq(NULL), rMaxSq_1_4(NULL) {}
  ~FF_EXP6() override {
#ifdef GOMC_CUDA
    DestroyExp6CUDAVars(getCUDAVars());
#endif
    delete[] expConst;
    delete[] expConst_1_4;
    delete[] rMin;
    delete[] rMin_1_4;
    delete[] rMaxSq;
    delete[] rMaxSq_1_4;
  }

  //! Exp-6 turns over at small r, so it is guarded by a hard wall. FFAdapter
  //! has no notion of that, so these two add it and then defer.
  double CalcEn(const double distSq, const uint kind1, const uint kind2,
                const double lambda) const override {
    if (distSq < rMaxSq[FlatIndex(kind1, kind2)])
      return num::BIGNUM;
    return FFAdapter::CalcEn(distSq, kind1, kind2, lambda);
  }
  double CalcVir(const double distSq, const uint kind1, const uint kind2,
                 const double lambda) const override {
    if (distSq < rMaxSq[FlatIndex(kind1, kind2)])
      return num::BIGNUM;
    return FFAdapter::CalcVir(distSq, kind1, kind2, lambda);
  }

  //! Parameter views; these add the Exp-6 specific arrays.
  ff::VdwParams VdwView() const {
    ff::VdwParams p = FFParticle::VdwView();
    p.expConst = expConst; p.rMin = rMin;
    return p;
  }
  ff::VdwParams VdwView14() const {
    ff::VdwParams p = FFParticle::VdwView14();
    p.expConst = expConst_1_4; p.rMin = rMin_1_4;
    return p;
  }

  void Init(ff_setup::Particle const &mie,
            ff_setup::NBfix const &nbfix) override;

  double GetRmin(const uint i, const uint j) const override;
  double GetRmax(const uint i, const uint j) const override;
  double GetRmin_1_4(const uint i, const uint j) const override;
  double GetRmax_1_4(const uint i, const uint j) const override;


  // coulomb interaction functions

  //! Returns energy correction
  double EnergyLRC(const uint kind1, const uint kind2) const override;

  //!!Returns virial correction
  double VirialLRC(const uint kind1, const uint kind2) const override;

  //! Returns impulse pressure correction term for a kind pair
  double ImpulsePressureCorrection(const uint kind1,
                                   const uint kind2) const override;

  // Calculate the dE/dlambda for vdw energy
  // Calculate the dE/dlambda for Coulomb energy

  double *expConst, *expConst_1_4, *rMin, *rMin_1_4, *rMaxSq, *rMaxSq_1_4;



protected:
};

inline void FF_EXP6::Init(ff_setup::Particle const &mie,
                          ff_setup::NBfix const &nbfix) {
  // Initialize sigma and epsilon
  FFParticle::Init(mie, nbfix);
  uint size = num::Sq(count);
  // allocate memory
  expConst = new double[size];
  rMin = new double[size];
  rMaxSq = new double[size];
  expConst_1_4 = new double[size];
  rMin_1_4 = new double[size];
  rMaxSq_1_4 = new double[size];
  // calculate exp-6 parameter
  for (uint i = 0; i < count; ++i) {
    for (uint j = 0; j < count; ++j) {
      uint idx = FlatIndex(i, j);
      // We use n as alpha for exp-6, with geometric combining
      expConst[idx] = epsilon[idx] * n[idx] / (n[idx] - 6.0);
      expConst_1_4[idx] = epsilon_1_4[idx] * n_1_4[idx] / (n_1_4[idx] - 6.0);

      // Find the Rmin(well depth)
      double sigma = sqrt(sigmaSq[idx]);
      num::Exp6Fun *func1 = new num::RminFun(n[idx], sigma);
      rMin[idx] = num::Zbrent(func1, sigma, 3.0 * sigma, 1.0e-7);
      // Find the Rmax(du/dr = 0)
      num::Exp6Fun *func2 = new num::RmaxFun(n[idx], sigma, rMin[idx]);
      double rmax = Zbrent(func2, 0.0, sigma, 1.0e-7);
      rMaxSq[idx] = rmax * rmax;

      // Find the Rmin(well depth)
      double sigma_1_4 = sqrt(sigmaSq_1_4[idx]);
      num::Exp6Fun *func3 = new num::RminFun(n_1_4[idx], sigma_1_4);
      rMin_1_4[idx] = num::Zbrent(func3, sigma_1_4, 3.0 * sigma_1_4, 1.0e-7);
      // Find the Rmax(du/dr = 0)
      num::Exp6Fun *func4 =
          new num::RmaxFun(n_1_4[idx], sigma_1_4, rMin_1_4[idx]);
      double rmax_1_4 = Zbrent(func4, 0.0, sigma_1_4, 1.0e-7);
      rMaxSq_1_4[idx] = rmax_1_4 * rmax_1_4;

      delete func1;
      delete func2;
      delete func3;
      delete func4;
    }
  }
#ifdef GOMC_CUDA
  InitExp6VariablesCUDA(getCUDAVars(), rMin, expConst, rMaxSq, size);
#endif
}











inline double FF_EXP6::GetRmin(const uint i, const uint j) const {
  uint idx = FlatIndex(i, j);
  return rMin[idx];
}

inline double FF_EXP6::GetRmax(const uint i, const uint j) const {
  uint idx = FlatIndex(i, j);
  return sqrt(rMaxSq[idx]);
}

inline double FF_EXP6::GetRmin_1_4(const uint i, const uint j) const {
  uint idx = FlatIndex(i, j);
  return rMin_1_4[idx];
}

inline double FF_EXP6::GetRmax_1_4(const uint i, const uint j) const {
  uint idx = FlatIndex(i, j);
  return sqrt(rMaxSq_1_4[idx]);
}

//! Returns energy correction
inline double FF_EXP6::EnergyLRC(const uint kind1, const uint kind2) const {
  uint idx = FlatIndex(kind1, kind2);
  double rCut = forcefield.rCut;
  // A,B and C for energy equation
  double A = 6.0 * epsilon[idx] * exp(n[idx]) / ((double)n[idx] - 6.0);
  double B = rMin[idx] / (double)n[idx];
  double C = epsilon[idx] * (double)n[idx] * pow(rMin[idx], 6.0) /
             ((double)n[idx] - 6.0);

  double tailCorrection =
      2.0 * M_PI *
      (A * B * exp(-rCut / B) * (2.0 * B * B + (2.0 * B * rCut) + rCut * rCut) -
       C / (3.0 * rCut * rCut * rCut));
  return tailCorrection;
}
//!!Returns virial correction
inline double FF_EXP6::VirialLRC(const uint kind1, const uint kind2) const {
  uint idx = FlatIndex(kind1, kind2);
  double rCut = forcefield.rCut;
  // A,B and C for virial equation
  double A = 6.0 * epsilon[idx] * exp(n[idx]) / ((double)n[idx] - 6.0);
  double B = rMin[idx] / (double)n[idx];
  double C = epsilon[idx] * (double)n[idx] * pow(rMin[idx], 6.0) /
             ((double)n[idx] - 6.0);
  double tailCorrection = 2.0 * M_PI *
                          (A * exp(-rCut / B) *
                               (6.0 * B * B * B + 6.0 * B * B * rCut +
                                3.0 * rCut * rCut * B + rCut * rCut * rCut) -
                           2.0 * C / (rCut * rCut * rCut));
  return tailCorrection;
}

inline double FF_EXP6::ImpulsePressureCorrection(const uint kind1,
                                                 const uint kind2) const {
  uint idx = FlatIndex(kind1, kind2);
  double rCut = forcefield.rCut;
  double rCut3 = rCut * forcefield.rCutSq;
  double A = 6.0 * epsilon[idx] * exp(n[idx]) / ((double)n[idx] - 6.0);
  double B = -1.0 * (double)n[idx] / rMin[idx];
  double C = epsilon[idx] * (double)n[idx] * pow(rMin[idx], 6.0) /
             ((double)n[idx] - 6.0);
  double tailCorrection = 2.0 * M_PI * (A * exp(rCut * B) * rCut3 - C / rCut3);
  return tailCorrection;
}


// Calculate the dE/dlambda for Coulomb energy

#endif /*FF_EXP6_H*/
