/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) 2.80
Copyright (C) 2022 GOMC Group
A copy of the MIT License can be found in License.txt
along with this program, also can be found at
<https://opensource.org/licenses/MIT>.
********************************************************************************/
#ifndef FF_ADAPTER_H
#define FF_ADAPTER_H

#include "CoulEvaluators.h"
#include "FFParticle.h"
#include "VdwEvaluators.h"

//
// One implementation of the forcefield interface, parameterised on which
// evaluators to use.
//
// Every kernel below -- cutoff guard, lambda handling, soft core, 1-4 handling
// -- is identical across forcefields; only the pair expressions differ, and
// those are the two template parameters. Before this, each forcefield carried
// its own ~150-line copy of all of it.
//
// Adding a forcefield is then:
//
//   struct FF_NEW final : public FFAdapter<FF_NEW, ff::MieNew, ff::PlainCoul> {
//     FF_NEW(Forcefield &ff) : FFAdapter(ff) {}
//     void Init(...) override;              // only if it has extra parameters
//     ff::VdwParams VdwView() const;        // only if it has extra parameters
//   };
//
// CRTP (the `Derived` parameter) rather than a virtual VdwView(): the parameter
// view is needed once per pair, so it has to resolve statically. Going through
// the vtable would put an indirect call back in the inner loop -- exactly what
// the templated energy kernels were changed to avoid.
//
// `Derived` must be marked `final` so the compiler can devirtualise the calls
// this class makes on itself.
//
template <class Derived, class VdwEval, class CoulEval>
struct FFAdapter : public FFParticle {
  explicit FFAdapter(Forcefield &ff) : FFParticle(ff) {}

  // ------------------------------------------------------ van der Waals ----

  double CalcEn(const double distSq, const uint kind1, const uint kind2,
                const double lambda) const override {
    if (forcefield.rCutSq < distSq)
      return 0.0;
    return ff::SoftCoreEnergy<VdwEval>(
        Self().VdwView(), distSq, FlatIndex(kind1, kind2), lambda,
        forcefield.sc_alpha, forcefield.sc_power, forcefield.sc_sigma_6);
  }

  double CalcVir(const double distSq, const uint kind1, const uint kind2,
                 const double lambda) const override {
    if (forcefield.rCutSq < distSq)
      return 0.0;
    return ff::SoftCoreVirial<VdwEval>(
        Self().VdwView(), distSq, FlatIndex(kind1, kind2), lambda,
        forcefield.sc_alpha, forcefield.sc_power, forcefield.sc_sigma_6);
  }

  void CalcAdd_1_4(double &en, const double distSq, const uint kind1,
                   const uint kind2) const override {
    if (forcefield.rCutSq < distSq)
      return;
    en += VdwEval::Energy(Self().VdwView14(), distSq, FlatIndex(kind1, kind2));
  }

  // ----------------------------------------------------- electrostatics ----

  double CalcCoulomb(const double distSq, const uint kind1, const uint kind2,
                     const double qi_qj_Fact, const double lambda,
                     const uint b) const override {
    if (forcefield.rCutCoulombSq[b] < distSq)
      return 0.0;
    if (lambda >= 0.999999)
      return CoulKernel(distSq, qi_qj_Fact, b);

    if (forcefield.sc_coul) {
      const ff::SoftCoreDist sc = ff::SoftenedDistance(
          distSq, sigmaSq[FlatIndex(kind1, kind2)], lambda, forcefield.sc_alpha,
          forcefield.sc_power, forcefield.sc_sigma_6);
      return lambda * CoulKernel(sc.softRsq, qi_qj_Fact, b);
    }
    return lambda * CoulKernel(distSq, qi_qj_Fact, b);
  }

  double CalcCoulombVir(const double distSq, const uint kind1,
                        const uint kind2, const double qi_qj,
                        const double lambda, const uint b) const override {
    if (forcefield.rCutCoulombSq[b] < distSq)
      return 0.0;
    if (lambda >= 0.999999)
      return CoulVirKernel(distSq, qi_qj, b);

    if (forcefield.sc_coul) {
      const ff::SoftCoreDist sc = ff::SoftenedDistance(
          distSq, sigmaSq[FlatIndex(kind1, kind2)], lambda, forcefield.sc_alpha,
          forcefield.sc_power, forcefield.sc_sigma_6);
      const double correction = distSq / sc.softRsq;
      return lambda * correction * correction *
             CoulVirKernel(sc.softRsq, qi_qj, b);
    }
    return lambda * CoulVirKernel(distSq, qi_qj, b);
  }

  void CalcCoulombAdd_1_4(double &en, const double distSq,
                          const double qi_qj_Fact,
                          const bool NB) const override {
    if (forcefield.rCutSq < distSq)
      return;
    const double dist = sqrt(distSq);
    en += NB ? qi_qj_Fact / dist : qi_qj_Fact * forcefield.scaling_14 / dist;
  }

  // -------------------------------------------------------- free energy ----

  double CalcdEndL(const double distSq, const uint kind1, const uint kind2,
                   const double lambda) const override {
    if (forcefield.rCutSq < distSq)
      return 0.0;
    return ff::DEnergyDLambda<VdwEval>(
        Self().VdwView(), distSq, FlatIndex(kind1, kind2), lambda,
        forcefield.sc_alpha, forcefield.sc_power, forcefield.sc_sigma_6);
  }

  double CalcCoulombdEndL(const double distSq, const uint kind1,
                          const uint kind2, const double qi_qj_Fact,
                          const double lambda, uint b) const override {
    if (forcefield.rCutCoulombSq[b] < distSq)
      return 0.0;
    if (!forcefield.sc_coul)
      return CoulKernel(distSq, qi_qj_Fact, b);

    const ff::SoftCoreDist sc = ff::SoftenedDistance(
        distSq, sigmaSq[FlatIndex(kind1, kind2)], lambda, forcefield.sc_alpha,
        forcefield.sc_power, forcefield.sc_sigma_6);
    double fCoef = lambda * forcefield.sc_alpha * forcefield.sc_power / 6.0;
    fCoef *= pow(1.0 - lambda, forcefield.sc_power - 1.0) * sc.sigma6 /
             (sc.softRsq * sc.softRsq);
    return CoulKernel(sc.softRsq, qi_qj_Fact, b) +
           fCoef * CoulVirKernel(sc.softRsq, qi_qj_Fact, b);
  }

protected:
  //! Static downcast; see the CRTP note above.
  const Derived &Self() const { return *static_cast<const Derived *>(this); }

  //! Ewald when it is on, this forcefield's own form otherwise.
  double CoulKernel(const double distSq, const double qi_qj_Fact,
                    const uint b) const {
    if (forcefield.ewald)
      return ff::EwaldReal::Energy(Self().CoulView(b), distSq, qi_qj_Fact);
    return CoulEval::Energy(Self().CoulView(b), distSq, qi_qj_Fact);
  }
  double CoulVirKernel(const double distSq, const double qi_qj,
                       const uint b) const {
    if (forcefield.ewald)
      return ff::EwaldReal::Virial(Self().CoulView(b), distSq, qi_qj);
    return CoulEval::Virial(Self().CoulView(b), distSq, qi_qj);
  }

  // The parameter-indexed forms the base class declares.
  double CalcEn(const double distSq, const uint index) const override {
    return VdwEval::Energy(Self().VdwView(), distSq, index);
  }
  double CalcVir(const double distSq, const uint index) const override {
    return VdwEval::Virial(Self().VdwView(), distSq, index);
  }
  double CalcCoulomb(const double distSq, const double qi_qj_Fact,
                     const uint b) const override {
    return CoulKernel(distSq, qi_qj_Fact, b);
  }
  double CalcCoulombVir(const double distSq, const double qi_qj,
                        uint b) const override {
    return CoulVirKernel(distSq, qi_qj, b);
  }
};

#endif /*FF_ADAPTER_H*/
