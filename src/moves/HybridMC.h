/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#ifndef HYBRIDMC_H
#define HYBRIDMC_H

#include <cmath>
#include <vector>

#include "MoveBase.h"
#include "RigidBody.h"
#include "System.h"

//
// Rigid-body hybrid Monte Carlo (rigid_body_hmc_design.md).
//
// Every mobile molecule in the chosen box gets Maxwell-Boltzmann momenta and
// the box is integrated for L steps of rigid-body dynamics (RigidBody.h);
// the end point is accepted with min(1, exp(-beta dH)). The integrator is
// volume-preserving and time-reversible, so the move is exact for any time
// step and trajectory length; integration error only lowers the acceptance.
//
// Molecules are rigid for the length of a trajectory. Over such a move the
// intramolecular, Ewald self and correction, and tail-correction energies are
// constant, so dU is the change in intermolecular LJ, real-space and
// reciprocal-space energy. Both end points are computed from scratch with the
// same functions (BoxForce, BoxReciprocal), so dU does not pick up any drift
// in the running energy total, and an accepted move resynchronizes it.
//
// The forces are an exact function of the positions (the cell list is
// rebuilt for every step), which is what reversibility requires.
//
class HybridMC : public MoveBase {
public:
  HybridMC(System &sys, StaticVals const &statV);
  ~HybridMC() {}

  virtual uint Prep(const double subDraw, const double movPerc);
  virtual uint PrepNEMTMC(const uint box, const uint midx = 0,
                          const uint kidx = 0);
  virtual uint Transform();
  virtual void CalcEn();
  virtual void Accept(const uint rejectState, const ulong step);
  virtual void PrintAcceptKind();

private:
  uint PrepBox();
  // Energy and forces of the box at newMolsPos; returns the changing part of
  // the energy (intermolecular LJ + real + reciprocal)
  double EvaluateForces(SystemPotential &pot);
  void KickAll(const double h);
  void DriftAll(const double h);
  double KineticAll() const;
  uint ChooseBox();

  uint bPick = 0;
  double kT;
  std::vector<uint> mobile;         // molecule indices that move
  std::vector<rigid::Body> bodies;  // one per mobile molecule
  std::vector<uint> atomStart;      // first atom of each mobile molecule
  std::vector<XYZ> atomForceTmp;    // per-atom total force, one molecule

  Coordinates newMolsPos;
  COM newCOMs;
  XYZArray atomForceRecNew, molForceRecNew;
  SystemPotential sysPotNew;
  double dH = 0.0;
  bool trajectoryOK = true;

  // settings
  double timeStep[BOX_TOTAL]; // fs
  uint stepsMin, stepsMax;
  bool adjust, useLiquid, useGas;
  ulong equilSteps;

  // statistics
  ulong intervalTries[BOX_TOTAL], intervalAccepted[BOX_TOTAL];
  double sumExpDH[BOX_TOTAL], sumDH[BOX_TOTAL];
  ulong totalTries[BOX_TOTAL];

  const MoleculeLookup &molLookup;
};

inline HybridMC::HybridMC(System &sys, StaticVals const &statV)
    : MoveBase(sys, statV),
      newMolsPos(sys.boxDimRef, newCOMs, sys.molLookupRef, sys.prng, statV.mol),
      newCOMs(sys.boxDimRef, newMolsPos, sys.molLookupRef, statV.mol),
      molLookup(sys.molLookup) {
  kT = 1.0 / statV.forcefield.beta;
  newMolsPos.Init(sys.coordinates.Count());
  newCOMs.Init(sys.com.Count());
  atomForceRecNew.Init(sys.coordinates.Count());
  molForceRecNew.Init(sys.com.Count());
  stepsMin = statV.hmcStepsMin;
  stepsMax = statV.hmcStepsMax;
  adjust = statV.hmcAdjust;
  useLiquid = statV.hmcLiquid;
  useGas = statV.hmcGas;
  equilSteps = statV.simEventFreq.tillEquil;
  for (uint b = 0; b < BOX_TOTAL; ++b) {
    timeStep[b] = statV.hmcTimeStep;
    intervalTries[b] = intervalAccepted[b] = totalTries[b] = 0;
    sumExpDH[b] = sumDH[b] = 0.0;
  }
}

// GEMC: the liquid (densest) or gas box when only one is enabled, as for the
// multiparticle move
inline uint HybridMC::ChooseBox() {
  double minDens = 1.0e20, maxDens = 0.0;
  uint minB = 0, maxB = 0;
  for (uint b = 0; b < BOX_TOTAL; ++b) {
    double density = 0.0;
    for (uint k = 0; k < molLookup.GetNumKind(); ++k)
      density += molLookup.NumKindInBox(k, b) * boxDimRef.volInv[b] *
                 molRef.kinds[k].molMass;
    if (density > maxDens) {
      maxDens = density;
      maxB = b;
    }
    if (density < minDens) {
      minDens = density;
      minB = b;
    }
  }
  return useLiquid ? maxB : minB;
}

inline uint HybridMC::Prep(const double subDraw, const double movPerc) {
#if ENSEMBLE == GCMC
  bPick = mv::BOX0;
#elif ENSEMBLE == GEMC
  if (useLiquid != useGas)
    bPick = ChooseBox();
  else
    prng.PickBox(bPick, subDraw, movPerc);
#else
  prng.PickBox(bPick, subDraw, movPerc);
#endif
  return PrepBox();
}

inline uint HybridMC::PrepNEMTMC(const uint box, const uint midx,
                                 const uint kidx) {
  bPick = box;
  return PrepBox();
}

// Body frames, momenta, and the forces and energy at the start
inline uint HybridMC::PrepBox() {
  mobile.clear();
  MoleculeLookup::box_iterator m = molLookup.BoxBegin(bPick);
  MoleculeLookup::box_iterator end = molLookup.BoxEnd(bPick);
  for (; m != end; ++m)
    if (!molLookup.IsFix(*m))
      mobile.push_back(*m);
  if (mobile.empty())
    return mv::fail_state::NO_MOL_OF_KIND_IN_BOX;

  coordCurrRef.CopyRange(newMolsPos, 0, 0, coordCurrRef.Count());
  comCurrRef.CopyRange(newCOMs, 0, 0, comCurrRef.Count());

  bodies.resize(mobile.size());
  atomStart.resize(mobile.size());
  std::vector<XYZ> pos;
  auto normal = [this]() { return prng.Gaussian(0.0, 1.0); };
  for (size_t i = 0; i < mobile.size(); ++i) {
    uint start, stop, len;
    molRef.GetRange(start, stop, len, mobile[i]);
    atomStart[i] = start;
    // whole molecule, unwrapped around its first atom
    XYZArray temp(len);
    coordCurrRef.CopyRange(temp, start, 0, len);
    boxDimRef.UnwrapPBC(temp, bPick, temp.Get(0));
    pos.resize(len);
    for (uint a = 0; a < len; ++a)
      pos[a] = temp.Get(a);
    const MoleculeKind &kind = molRef.GetKind(mobile[i]);
    bodies[i].Setup(pos.data(), kind.atomMass, len);
    bodies[i].DrawMomenta(kT, normal);
  }
  return mv::fail_state::NO_FAIL;
}

inline double HybridMC::EvaluateForces(SystemPotential &pot) {
  cellList.GridAll(boxDimRef, newMolsPos, molLookup);
  calcEwald->BoxReciprocalSums(bPick, newMolsPos);
  pot = calcEnRef.BoxForce(sysPotRef, newMolsPos, atomForceNew, molForceNew,
                           boxDimRef, bPick);
  pot.boxEnergy[bPick].recip = calcEwald->BoxReciprocal(bPick, false);
  calcEwald->BoxForceReciprocal(newMolsPos, atomForceRecNew, molForceRecNew,
                                bPick);
  const Energy &e = pot.boxEnergy[bPick];
  return e.inter + e.real + e.recip;
}

inline void HybridMC::KickAll(const double h) {
  for (size_t i = 0; i < bodies.size(); ++i) {
    rigid::Body &b = bodies[i];
    atomForceTmp.resize(b.NumAtoms());
    for (uint a = 0; a < b.NumAtoms(); ++a) {
      const uint p = atomStart[i] + a;
      atomForceTmp[a] = atomForceNew.Get(p) + atomForceRecNew.Get(p);
    }
    XYZ force, torque;
    rigid::ForceAndTorque(b, atomForceTmp.data(), force, torque);
    b.Kick(force, torque, h);
  }
}

inline void HybridMC::DriftAll(const double h) {
  for (size_t i = 0; i < bodies.size(); ++i) {
    rigid::Body &b = bodies[i];
    b.Drift(h);
    for (uint a = 0; a < b.NumAtoms(); ++a)
      newMolsPos.Set(atomStart[i] + a, boxDimRef.WrapPBC(b.Atom(a), bPick));
  }
}

inline double HybridMC::KineticAll() const {
  double k = 0.0;
  for (const rigid::Body &b : bodies)
    k += b.Kinetic();
  return k;
}

// The trajectory. Ends with the final energy in sysPotNew and dH set.
inline uint HybridMC::Transform() {
  // back up the cached Fourier terms (EwaldCached); restored on rejection
  calcEwald->backupMolCache();
  SystemPotential pot;
  const double u0 = EvaluateForces(pot);
  const double k0 = KineticAll();
  const uint L =
      stepsMin + (stepsMax > stepsMin ? prng.randIntExc(stepsMax - stepsMin + 1)
                                      : 0);
  const double h = rigid::FemtosecondToInternal(timeStep[bPick]);
  double u = u0;
  trajectoryOK = std::isfinite(u0);
  for (uint s = 0; s < L && trajectoryOK; ++s) {
    KickAll(0.5 * h);
    DriftAll(h);
    u = EvaluateForces(pot);
    // a non-finite energy is a property of the state, so stopping here and
    // rejecting keeps the move exact (the reverse path would stop too)
    if (!std::isfinite(u)) {
      trajectoryOK = false;
      break;
    }
    KickAll(0.5 * h);
  }
  sysPotNew = pot;
  dH = trajectoryOK ? (u - u0) + (KineticAll() - k0) : 0.0;
  return mv::fail_state::NO_FAIL;
}

inline void HybridMC::CalcEn() {}

inline void HybridMC::Accept(const uint rejectState, const ulong step) {
  bool result = false;
  if (rejectState == mv::fail_state::NO_FAIL && trajectoryOK) {
    const double w = exp(-BETA * dH);
    sumExpDH[bPick] += w;
    sumDH[bPick] += dH;
    ++totalTries[bPick];
    result = prng() < w;
  }
  if (result) {
    // the end point's energy was computed from scratch, so it also
    // resynchronizes the running total for this box
    sysPotRef = sysPotNew;
    sysPotRef.Total();
    swap(coordCurrRef, newMolsPos);
    for (uint m : mobile)
      comCurrRef.SetNew(m, bPick);
    calcEwald->UpdateRecip(bPick);
    velocity.UpdateBoxVelocity(bPick);
    // cell list already holds the new positions
  } else if (rejectState == mv::fail_state::NO_FAIL) {
    cellList.GridAll(boxDimRef, coordCurrRef, molLookup);
    calcEwald->exgMolCache();
  }

  // time-step adaptation, during equilibration only
  if (rejectState == mv::fail_state::NO_FAIL) {
    ++intervalTries[bPick];
    if (result)
      ++intervalAccepted[bPick];
    const ulong interval = 10;
    if (adjust && step < equilSteps && intervalTries[bPick] >= interval) {
      const double acc = double(intervalAccepted[bPick]) / intervalTries[bPick];
      const double target = 0.75;
      double f = std::pow((acc + 0.05) / (target + 0.05), 0.5);
      f = std::max(0.5, std::min(1.5, f));
      timeStep[bPick] *= f;
      intervalTries[bPick] = intervalAccepted[bPick] = 0;
    }
  }
  moveSetRef.Update(mv::HYBRID_MC, result, bPick);
}

inline void HybridMC::PrintAcceptKind() {
  printf("%-37s", "% Accepted Hybrid MC ");
  for (uint b = 0; b < BOX_TOTAL; b++)
    printf("%10.5f ", 100.0 * moveSetRef.GetAccept(b, mv::HYBRID_MC));
  std::cout << std::endl;
  printf("%-37s", "  Hybrid MC time step (fs) ");
  for (uint b = 0; b < BOX_TOTAL; b++)
    printf("%10.5f ", timeStep[b]);
  std::cout << std::endl;
  // <exp(-beta dH)> should be 1 within its error for a correct integrator
  printf("%-37s", "  Hybrid MC <exp(-beta dH)> ");
  for (uint b = 0; b < BOX_TOTAL; b++)
    printf("%10.5f ", totalTries[b] ? sumExpDH[b] / totalTries[b] : 0.0);
  std::cout << std::endl;
}

#endif /*HYBRIDMC_H*/
