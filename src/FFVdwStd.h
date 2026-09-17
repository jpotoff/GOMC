/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) 2.80
Copyright (C) 2022 GOMC Group
A copy of the MIT License can be found in License.txt
along with this program, also can be found at
<https://opensource.org/licenses/MIT>.
********************************************************************************/
#ifndef FF_VDW_STD_H
#define FF_VDW_STD_H

#include "FFAdapter.h"

//    Plain (truncated) Mie potential -- GOMC's VDW_STD_KIND.
//    U(rij) = cn * eps_ij * ( (sig_ij/rij)^n - (sig_ij/rij)^6)
//    cn = n/(n-6) * ((n/6)^(6/(n-6)))
//
//    Nothing is done to the energy at the cutoff, so unlike the shifted and
//    switched forms this one needs real long-range corrections; those are
//    FFParticle's own EnergyLRC/VirialLRC/ImpulsePressureCorrection, inherited
//    unchanged. The pair expressions are ff::MiePlain and ff::PlainCoul.
//
//    This used to be FFParticle itself: the base class was also the concrete
//    VDW_STD forcefield. That is why FFParticle could not be marked `final`
//    and why its pair kernels needed a separate FFParticleInline.h (now gone).
//    Splitting
//    it out leaves FFParticle as parameter storage plus the interface, and
//    makes all five concrete forcefields `final`.
//
struct FF_VDW_STD final
    : public FFAdapter<FF_VDW_STD, ff::MiePlain, ff::PlainCoul> {
  friend struct FFTestAccess;

public:
  explicit FF_VDW_STD(Forcefield &ff) : FFAdapter(ff) {}
};

#endif /*FF_VDW_STD_H*/
