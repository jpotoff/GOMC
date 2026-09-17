/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) 2.80
Copyright (C) 2022 GOMC Group
A copy of the MIT License can be found in License.txt
along with this program, also can be found at
<https://opensource.org/licenses/MIT>.
********************************************************************************/
#ifndef FORCEFIELD_DISPATCH_H
#define FORCEFIELD_DISPATCH_H

#include "ConfigSetup.h"
#include "FFExp6.h"
#include "FFParticle.h"
#include "FFShift.h"
#include "FFSwitch.h"
#include "FFSwitchMartini.h"
#include "FFVdwStd.h"
#include "Forcefield.h"

#include <type_traits>

//
// Resolve `forcefield.particles` to its concrete type once, so the energy
// kernels can be templated on it and call through it without the vtable.
//
// The concrete type is fixed at startup by vdwKind and isMartini (see
// Forcefield::InitBasicVals), so this costs one branch per kernel invocation
// rather than one indirect call per interacting pair.
//
// IMPORTANT: templating alone does not devirtualise. A `const FF_SHIFT&` may
// still refer to a further-derived object as far as the compiler knows, so an
// ordinary `ff.CalcEn(...)` remains an indirect call. Kernels must therefore
// use the qualified form, `ff.FFType::CalcEn(...)`, which names the function
// statically and suppresses dispatch. All five concrete forcefields are also
// `final`, which gives the compiler the same information a second way.
//
// Usage:
//   DispatchForcefield(forcefield, [&](const auto &ff) {
//     using FFT = std::decay_t<decltype(ff)>;
//     MyKernel<BoxDimensions, FFT>(ff, ...);
//   });
//
template <class Fn>
inline void DispatchForcefield(const Forcefield &forcefield, Fn &&fn) {
  typedef config_setup::FFValues FFV;
  const FFParticle &base = *forcefield.particles;

  if (forcefield.vdwKind == FFV::VDW_STD_KIND) {
    fn(static_cast<const FF_VDW_STD &>(base));
  } else if (forcefield.vdwKind == FFV::VDW_SHIFT_KIND) {
    fn(static_cast<const FF_SHIFT &>(base));
  } else if (forcefield.vdwKind == FFV::VDW_EXP6_KIND) {
    fn(static_cast<const FF_EXP6 &>(base));
  } else if (forcefield.isMartini) {
    fn(static_cast<const FF_SWITCH_MARTINI &>(base));
  } else {
    fn(static_cast<const FF_SWITCH &>(base));
  }
}

#endif /*FORCEFIELD_DISPATCH_H*/
