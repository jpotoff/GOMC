/*******************************************************************************
Tests for the rigid-body integrator used by hybrid Monte Carlo (RigidBody.h).

The integrator only has to be volume-preserving and time-reversible for HMC
to be exact, so those are tested directly (angular momentum conservation and
the energy error of the free rotor, reversal of a trajectory). Exactness is
then tested end to end: HMC on small systems whose equilibrium distributions
are known in closed form,
  - a neutral rigid molecule in a uniform field, where the dipole direction
    follows the Langevin distribution, <cos> = coth(a) - 1/a;
  - a harmonic tether on the centre of mass, <R^2> = 3 kT / k;
and the identity <exp(-beta dH)> = 1, which holds for any volume-preserving,
reversible integrator started from equilibrium. Asymmetric, linear and
single-atom bodies are covered.
*******************************************************************************/

#include <cmath>
#include <functional>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <vector>

#include "RigidBody.h"

namespace {

using rigid::Body;

const double kT = 300.0;

struct Rng {
  explicit Rng(unsigned seed) : gen(seed) {}
  double operator()() { return normal(gen); }
  double Uniform() { return uniform(gen); }
  std::mt19937_64 gen;
  std::normal_distribution<double> normal{0.0, 1.0};
  std::uniform_real_distribution<double> uniform{0.0, 1.0};
};

struct Molecule {
  std::vector<XYZ> pos;
  std::vector<double> mass, charge;
};

// Water-like asymmetric top, tilted so no principal axis is a lab axis
Molecule Asymmetric() {
  Molecule m;
  m.pos = {XYZ(0.10, 0.20, -0.05), XYZ(1.03, 0.42, 0.11),
           XYZ(-0.21, 1.08, 0.33)};
  m.mass = {15.999, 1.008, 1.008};
  m.charge = {-0.8, 0.4, 0.4};
  return m;
}

// Four atoms, all moments distinct and well separated
Molecule Branched() {
  Molecule m;
  m.pos = {XYZ(0.0, 0.0, 0.0), XYZ(1.5, 0.1, -0.2), XYZ(-0.4, 1.3, 0.3),
           XYZ(0.2, -0.5, 1.4)};
  m.mass = {12.011, 1.008, 14.007, 15.999};
  m.charge = {0.3, 0.1, -0.6, 0.2};
  return m;
}

Molecule Linear() {
  Molecule m;
  const XYZ axis = XYZ(0.3, -0.5, 0.81) * (1.0 / std::sqrt(0.09 + 0.25 + 0.6561));
  m.pos = {XYZ(0.5, 0.5, 0.5), XYZ(0.5, 0.5, 0.5) + axis * 1.13};
  m.mass = {12.011, 15.999};
  m.charge = {0.5, -0.5};
  return m;
}

Molecule SingleAtom() {
  Molecule m;
  m.pos = {XYZ(0.2, -0.1, 0.3)};
  m.mass = {39.948};
  m.charge = {0.0};
  return m;
}

Body MakeBody(const Molecule &m) {
  Body b;
  b.Setup(m.pos.data(), m.mass.data(), (uint)m.pos.size());
  return b;
}

// Energy and per-atom forces of a set of bodies
using ForceField = std::function<double(const std::vector<Body> &,
                                        std::vector<std::vector<XYZ>> &)>;

// Uniform field along z on atomic charges, plus a harmonic tether on each
// centre of mass (applied as mass-weighted atom forces, so it exerts no
// torque)
ForceField FieldAndTether(const std::vector<std::vector<double>> &charges,
                          double field, double k) {
  return [charges, field, k](const std::vector<Body> &bodies,
                             std::vector<std::vector<XYZ>> &f) {
    double u = 0.0;
    f.resize(bodies.size());
    for (size_t i = 0; i < bodies.size(); ++i) {
      const Body &b = bodies[i];
      f[i].assign(b.NumAtoms(), XYZ());
      u += 0.5 * k * b.com.LengthSq();
      for (uint a = 0; a < b.NumAtoms(); ++a) {
        const XYZ x = b.Atom(a);
        u -= charges[i][a] * field * x.z;
        f[i][a] = XYZ(0.0, 0.0, charges[i][a] * field) +
                  b.com * (-k * b.atomMass[a] / b.mass);
      }
    }
    return u;
  };
}

// Smooth Gaussian repulsion between atoms of different bodies, plus tethers
ForceField GaussianPairs(double amp, double width, double k) {
  return [amp, width, k](const std::vector<Body> &bodies,
                         std::vector<std::vector<XYZ>> &f) {
    double u = 0.0;
    f.resize(bodies.size());
    for (size_t i = 0; i < bodies.size(); ++i) {
      f[i].assign(bodies[i].NumAtoms(), XYZ());
      u += 0.5 * k * bodies[i].com.LengthSq();
      for (uint a = 0; a < bodies[i].NumAtoms(); ++a)
        f[i][a] += bodies[i].com *
                   (-k * bodies[i].atomMass[a] / bodies[i].mass);
    }
    const double w2 = width * width;
    for (size_t i = 0; i < bodies.size(); ++i)
      for (size_t j = i + 1; j < bodies.size(); ++j)
        for (uint a = 0; a < bodies[i].NumAtoms(); ++a)
          for (uint c = 0; c < bodies[j].NumAtoms(); ++c) {
            const XYZ d = bodies[i].Atom(a) - bodies[j].Atom(c);
            const double e = amp * std::exp(-d.LengthSq() / w2);
            u += e;
            const XYZ fij = d * (2.0 * e / w2);
            f[i][a] += fij;
            f[j][c] -= fij;
          }
    return u;
  };
}

double Kinetic(const std::vector<Body> &bodies, bool rotational = true) {
  double k = 0.0;
  for (const Body &b : bodies)
    k += rotational ? b.Kinetic() : 0.5 * b.mom.LengthSq() / b.mass;
  return k;
}

// Negative controls. Irreversible: each step is a full kick then a drift
// (symplectic Euler), volume-preserving but not time-reversible; its bias is
// real but small for these systems. NoRotationalEnergy: the Metropolis test
// leaves the rotational kinetic energy out of H, a plausible accounting error
// that biases the orientation strongly. (Turning Q while leaving pi
// unrotated is NOT a control: that map is still a volume-preserving
// involution, so HMC stays exact with the wrong dynamics.)
enum class Scheme { Exact, Irreversible, NoRotationalEnergy };

// L velocity-Verlet steps; f holds the forces at the start and is left with
// the forces at the end. Returns the final potential energy.
double Integrate(std::vector<Body> &bodies, const ForceField &ff,
                 std::vector<std::vector<XYZ>> &f, double h, uint steps,
                 Scheme scheme = Scheme::Exact) {
  const bool halfKicks = scheme != Scheme::Irreversible;
  double u = 0.0;
  for (uint s = 0; s < steps; ++s) {
    for (size_t i = 0; i < bodies.size(); ++i) {
      XYZ force, torque;
      rigid::ForceAndTorque(bodies[i], f[i].data(), force, torque);
      bodies[i].Kick(force, torque, halfKicks ? 0.5 * h : h);
      bodies[i].Drift(h);
    }
    u = ff(bodies, f);
    if (!halfKicks)
      continue;
    for (size_t i = 0; i < bodies.size(); ++i) {
      XYZ force, torque;
      rigid::ForceAndTorque(bodies[i], f[i].data(), force, torque);
      bodies[i].Kick(force, torque, 0.5 * h);
    }
  }
  return u;
}

struct HmcStats {
  double acceptance = 0.0;
  double meanExpDH = 0.0;
  uint moves = 0;
};

// Full-momentum-refresh HMC. observe(bodies) is called after every move.
HmcStats RunHmc(std::vector<Body> &bodies, const ForceField &ff, double h,
                uint lMin, uint lMax, uint moves, Rng &rng,
                const std::function<void(const std::vector<Body> &)> &observe,
                Scheme scheme = Scheme::Exact) {
  HmcStats st;
  std::vector<std::vector<XYZ>> f;
  double u = ff(bodies, f);
  uint accepted = 0;
  double sumExp = 0.0;
  for (uint m = 0; m < moves; ++m) {
    const std::vector<Body> saved = bodies;
    const std::vector<std::vector<XYZ>> savedF = f;
    const double savedU = u;
    for (Body &b : bodies)
      b.DrawMomenta(kT, rng);
    const bool rot = scheme != Scheme::NoRotationalEnergy;
    const double h0 = u + Kinetic(bodies, rot);
    const uint L = lMin + (uint)(rng.Uniform() * (lMax - lMin + 1));
    const double u1 = Integrate(bodies, ff, f, h, L, scheme);
    const double dH = u1 + Kinetic(bodies, rot) - h0;
    const double w = std::exp(-dH / kT);
    sumExp += w;
    if (rng.Uniform() < w) {
      u = u1;
      ++accepted;
    } else {
      bodies = saved;
      f = savedF;
      u = savedU;
    }
    observe(bodies);
  }
  st.moves = moves;
  st.acceptance = double(accepted) / moves;
  st.meanExpDH = sumExp / moves;
  return st;
}

double Langevin(double a) { return 1.0 / std::tanh(a) - 1.0 / a; }

XYZ Dipole(const Body &b, const std::vector<double> &q) {
  XYZ mu;
  for (uint a = 0; a < b.NumAtoms(); ++a)
    mu += b.ToSpace(b.body[a]) * q[a];
  return mu;
}

} // namespace

TEST(RigidBodyHMCTest, TimeUnit) {
  // A sqrt(amu / K) = 1.0967 ps
  EXPECT_NEAR(rigid::FemtosecondToInternal(1.0), 9.1183e-4, 1e-7);
}

TEST(RigidBodyHMCTest, BodyFrame) {
  for (const Molecule &m : {Asymmetric(), Branched(), Linear(), SingleAtom()}) {
    Body b = MakeBody(m);
    // atoms reconstructed
    for (uint a = 0; a < b.NumAtoms(); ++a) {
      const XYZ d = b.Atom(a) - m.pos[a];
      EXPECT_LT(d.Length(), 1e-12);
    }
    // Q orthonormal with determinant +1
    for (uint i = 0; i < 3; ++i)
      for (uint j = 0; j < 3; ++j) {
        double dot = 0.0;
        for (uint r = 0; r < 3; ++r)
          dot += b.Q[r][i] * b.Q[r][j];
        EXPECT_NEAR(dot, i == j ? 1.0 : 0.0, 1e-12);
      }
    const double det =
        b.Q[0][0] * (b.Q[1][1] * b.Q[2][2] - b.Q[1][2] * b.Q[2][1]) -
        b.Q[0][1] * (b.Q[1][0] * b.Q[2][2] - b.Q[1][2] * b.Q[2][0]) +
        b.Q[0][2] * (b.Q[1][0] * b.Q[2][1] - b.Q[1][1] * b.Q[2][0]);
    EXPECT_NEAR(det, 1.0, 1e-12);
    // inertia tensor diagonal in the body frame, with the stored moments
    double t[3][3] = {{0.0}};
    for (uint a = 0; a < b.NumAtoms(); ++a) {
      const double v[3] = {b.body[a].x, b.body[a].y, b.body[a].z};
      const double r2 = b.body[a].LengthSq();
      for (uint i = 0; i < 3; ++i)
        for (uint j = 0; j < 3; ++j)
          t[i][j] += b.atomMass[a] * ((i == j ? r2 : 0.0) - v[i] * v[j]);
    }
    for (uint i = 0; i < 3; ++i)
      for (uint j = 0; j < 3; ++j)
        EXPECT_NEAR(t[i][j], i == j ? b.I[i] : 0.0, 1e-10);
  }
  EXPECT_EQ(MakeBody(Linear()).active[0] + MakeBody(Linear()).active[1] +
                MakeBody(Linear()).active[2],
            2);
  const Body atom = MakeBody(SingleAtom());
  EXPECT_FALSE(atom.active[0] || atom.active[1] || atom.active[2]);
}

// Free asymmetric rotor: the splitting conserves the space-frame angular
// momentum exactly (each sub-step does), and the energy error is O(h^2) and
// bounded, with no drift.
TEST(RigidBodyHMCTest, FreeRotor) {
  auto maxEnergyError = [](double fs, uint steps, double &lErr, double &late,
                           double &early) {
    Body b = MakeBody(Branched());
    Rng rng(11);
    b.DrawMomenta(kT, rng);
    b.mom = XYZ();
    const double e0 = b.Kinetic();
    const XYZ l0 = b.SpaceAngularMomentum();
    const double h = rigid::FemtosecondToInternal(fs);
    double maxErr = 0.0;
    early = late = 0.0;
    lErr = 0.0;
    for (uint s = 0; s < steps; ++s) {
      b.Drift(h);
      const double err = std::abs(b.Kinetic() - e0) / e0;
      maxErr = std::max(maxErr, err);
      if (s < steps / 10)
        early = std::max(early, err);
      if (s >= steps - steps / 10)
        late = std::max(late, err);
      lErr = std::max(lErr, (b.SpaceAngularMomentum() - l0).Length() /
                                l0.Length());
    }
    return maxErr;
  };
  double lErr, late, early;
  const double e2 = maxEnergyError(2.0, 200000, lErr, late, early);
  EXPECT_LT(lErr, 1e-10);
  EXPECT_LT(e2, 1e-2);
  // bounded: the error in the last tenth is no larger than in the first
  EXPECT_LT(late, 2.0 * early);
  double lErr1, late1, early1;
  const double e1 = maxEnergyError(1.0, 400000, lErr1, late1, early1);
  // second order: halving the step cuts the error by about four
  EXPECT_GT(e2 / e1, 3.0);
  EXPECT_LT(e2 / e1, 5.0);
}

// Integrate forward, flip the momenta, integrate back: the start comes back
TEST(RigidBodyHMCTest, Reversible) {
  std::vector<Body> bodies;
  const XYZ offsets[3] = {XYZ(0, 0, 0), XYZ(2.4, 0.3, -0.2),
                          XYZ(-0.5, 2.6, 0.4)};
  for (uint i = 0; i < 3; ++i) {
    Molecule m = (i == 1) ? Linear() : Branched();
    for (XYZ &p : m.pos)
      p += offsets[i];
    bodies.push_back(MakeBody(m));
  }
  Rng rng(5);
  for (Body &b : bodies)
    b.DrawMomenta(kT, rng);
  const std::vector<Body> start = bodies;
  const ForceField ff = GaussianPairs(800.0, 1.5, 20.0);
  std::vector<std::vector<XYZ>> f;
  const double u0 = ff(bodies, f);
  const double h = rigid::FemtosecondToInternal(2.0);
  Integrate(bodies, ff, f, h, 300);
  for (Body &b : bodies)
    b.FlipMomenta();
  Integrate(bodies, ff, f, h, 300);
  double maxDev = 0.0;
  for (size_t i = 0; i < bodies.size(); ++i)
    for (uint a = 0; a < bodies[i].NumAtoms(); ++a)
      maxDev = std::max(maxDev,
                        (bodies[i].Atom(a) - start[i].Atom(a)).Length());
  EXPECT_LT(maxDev, 1e-9);
  EXPECT_NEAR(ff(bodies, f), u0, 1e-8 * std::abs(u0) + 1e-9);
}

// HMC samples the right distribution: orientation (Langevin) and position
// (harmonic tether), at a time step large enough for a substantial rejection
// rate, where an inexact integrator would show its bias. <exp(-beta dH)> = 1
// is checked at a moderate step: at low acceptance dH has heavy tails and the
// sample mean of exp(-beta dH) is dominated by rare events.
struct FieldResult {
  HmcStats st;
  double cos = 0.0, cos2 = 0.0, r2 = 0.0;
  bool hasDipole = false;
};

const double kFieldA = 2.0;   // beta * field * |mu|
const double kTether = 300.0; // K / A^2

FieldResult RunField(const Molecule &m, unsigned seed, double fs,
                     Scheme scheme = Scheme::Exact, uint moves = 60000) {
  FieldResult r;
  std::vector<Body> bodies = {MakeBody(m)};
  const double muLen = Dipole(bodies[0], m.charge).Length();
  r.hasDipole = muLen > 0.0;
  const double field = r.hasDipole ? kFieldA * kT / muLen : 0.0;
  const ForceField ff = FieldAndTether({m.charge}, field, kTether);
  Rng rng(seed);
  double sumCos = 0.0, sumCos2 = 0.0, sumR2 = 0.0;
  uint n = 0;
  r.st = RunHmc(
      bodies, ff, rigid::FemtosecondToInternal(fs), 20, 60, moves, rng,
      [&](const std::vector<Body> &b) {
        if (r.hasDipole) {
          const double c = Dipole(b[0], m.charge).z / muLen;
          sumCos += c;
          sumCos2 += c * c;
        }
        sumR2 += b[0].com.LengthSq();
        ++n;
      },
      scheme);
  r.cos = sumCos / n;
  r.cos2 = sumCos2 / n;
  r.r2 = sumR2 / n;
  std::cout << "dt " << fs << " fs: acceptance " << r.st.acceptance
            << ", <exp(-beta dH)> " << r.st.meanExpDH << ", <cos> " << r.cos
            << " (exact " << Langevin(kFieldA) << "), <R^2> " << r.r2
            << " (exact " << 3.0 * kT / kTether << ")" << std::endl;
  return r;
}

// Time step (fs) at which HMC acceptance falls in [lo, hi], found with short
// runs. acceptance(fs) runs one.
double Calibrate(const std::function<double(double)> &acceptance, double lo,
                 double hi) {
  double fs = 10.0;
  for (uint it = 0; it < 60; ++it) {
    const double acc = acceptance(fs);
    if (acc > hi)
      fs *= 1.2;
    else if (acc < lo)
      fs /= 1.1;
    else
      return fs;
  }
  return fs;
}

double CalibrateField(const Molecule &m, unsigned seed, double lo = 0.6,
                      double hi = 0.9) {
  return Calibrate(
      [&](double fs) {
        return RunField(m, seed + 1000, fs, Scheme::Exact, 3000).st.acceptance;
      },
      lo, hi);
}

void CheckFieldSampling(const Molecule &m, unsigned seed) {
  const FieldResult r = RunField(m, seed, CalibrateField(m, seed));
  EXPECT_GT(r.st.acceptance, 0.5);
  EXPECT_LT(r.st.acceptance, 0.95);
  EXPECT_NEAR(r.r2, 3.0 * kT / kTether, 0.03 * 3.0 * kT / kTether);
  if (r.hasDipole) {
    EXPECT_NEAR(r.cos, Langevin(kFieldA), 0.015);
    EXPECT_NEAR(r.cos2, 1.0 - 2.0 * Langevin(kFieldA) / kFieldA, 0.015);
  }
  const FieldResult e =
      RunField(m, seed + 7, CalibrateField(m, seed + 7, 0.85, 0.97));
  EXPECT_NEAR(e.st.meanExpDH, 1.0, 0.02);
}

TEST(RigidBodyHMCTest, SamplesAsymmetricTop) {
  CheckFieldSampling(Branched(), 21);
}
TEST(RigidBodyHMCTest, SamplesWaterLike) {
  CheckFieldSampling(Asymmetric(), 22);
}
TEST(RigidBodyHMCTest, SamplesLinear) { CheckFieldSampling(Linear(), 23); }
TEST(RigidBodyHMCTest, SamplesSingleAtom) {
  CheckFieldSampling(SingleAtom(), 24);
}

// Negative control: an HMC that is not exact must fail the checks above, or
// they would not detect a broken implementation
TEST(RigidBodyHMCTest, BrokenHmcIsDetected) {
  const double fs = CalibrateField(Branched(), 21);
  const FieldResult r =
      RunField(Branched(), 21, fs, Scheme::NoRotationalEnergy);
  const bool biased =
      std::abs(r.cos - Langevin(kFieldA)) > 0.015 ||
      std::abs(r.cos2 - (1.0 - 2.0 * Langevin(kFieldA) / kFieldA)) > 0.015 ||
      std::abs(r.r2 - 3.0 * kT / kTether) > 0.03 * 3.0 * kT / kTether;
  EXPECT_TRUE(biased);
}

// Several interacting bodies: <exp(-beta dH)> = 1 still holds, at a time step
// where the energy error is not negligible
TEST(RigidBodyHMCTest, InteractingExpDH) {
  std::vector<Body> start;
  const XYZ offsets[4] = {XYZ(0, 0, 0), XYZ(2.5, 0, 0), XYZ(0, 2.5, 0),
                          XYZ(0, 0, 2.5)};
  for (uint i = 0; i < 4; ++i) {
    Molecule m = (i % 2) ? Asymmetric() : Branched();
    for (XYZ &p : m.pos)
      p += offsets[i];
    start.push_back(MakeBody(m));
  }
  const ForceField ff = GaussianPairs(800.0, 1.5, 20.0);
  auto run = [&](double fs, uint moves, unsigned seed) {
    std::vector<Body> bodies = start;
    Rng rng(seed);
    return RunHmc(bodies, ff, rigid::FemtosecondToInternal(fs), 20, 40, moves,
                  rng, [](const std::vector<Body> &) {});
  };
  const double fs = Calibrate(
      [&](double f) { return run(f, 2000, 131).acceptance; }, 0.85, 0.97);
  const HmcStats st = run(fs, 20000, 31);
  std::cout << "dt " << fs << " fs: acceptance " << st.acceptance
            << ", <exp(-beta dH)> " << st.meanExpDH << std::endl;
  EXPECT_GT(st.acceptance, 0.75);
  EXPECT_LT(st.acceptance, 0.99);
  EXPECT_NEAR(st.meanExpDH, 1.0, 0.02);
}
