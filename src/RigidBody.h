/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#ifndef RIGIDBODY_H
#define RIGIDBODY_H

#include <algorithm>
#include <cmath>
#include <vector>

#include "BasicTypes.h"
#include "GeomLib.h"

//
// Rigid-body dynamics for hybrid Monte Carlo (rigid_body_hmc_design.md).
//
// A molecule is held rigid for the length of an HMC trajectory: centre of
// mass R and momentum P, orientation Q (body -> space, columns are the
// principal axes), principal moments I_k and body-frame angular momentum pi,
// and body-frame atom positions b_a, so that atom a sits at R + Q b_a.
//
//   H = U(x) + |P|^2 / 2M + sum_k pi_k^2 / 2 I_k
//
// One step is the symmetric splitting
//
//   Kick(h/2)  Drift(h)  [forces at the new positions]  Kick(h/2)
//
// where Drift moves R and runs the free-rotor splitting of Dullweber,
// Leimkuhler and McLachlan (J. Chem. Phys. 107, 5840 (1997)):
//
//   R1(h/2) R2(h/2) R3(h) R2(h/2) R1(h/2)
//
// Each Rk is the exact flow of pi_k^2 / 2 I_k, a rotation of Q and pi about
// principal axis k, so the step is symplectic, and the symmetric order makes
// it time-reversible. That, and not accuracy, is what makes HMC exact.
//
// A linear molecule has two equal moments and none about its axis. The
// splitting does not apply there: a rotation about one perpendicular axis
// alone would give the axis itself angular momentum. Its free motion is
// instead integrated exactly, as one rotation of the molecule about the
// (constant) angular momentum L by h |L| / I. A single atom does not rotate.
//
// Units are GOMC's: angstrom, amu, kelvin (energies in K, so k_B = 1),
// forces in K/A and torques in K. The matching time unit is
// A sqrt(amu / K) = 1.0967 ps; FemtosecondToInternal converts.
//
namespace rigid {

//! Time step in internal units for a step of fs femtoseconds
inline double FemtosecondToInternal(const double fs) {
  // A sqrt(amu / (k_B K)) in femtoseconds
  const double unitFs = 1.0e5 * std::sqrt(1.66053906660e-27 / 1.380649e-23);
  return fs / unitFs;
}

struct Body {
  double mass = 0.0;
  XYZ com, mom;               // centre of mass and its momentum, space frame
  double Q[3][3];             // orientation, body -> space
  double I[3];                // principal moments
  bool active[3];             // axes with a nonzero moment
  bool linear = false;        // exactly two active axes, with equal moments
  XYZ pi;                     // angular momentum, body frame
  std::vector<XYZ> body;      // atom positions, body frame
  std::vector<double> atomMass;

  uint NumAtoms() const { return (uint)body.size(); }

  //! Space-frame position of atom a
  XYZ Atom(const uint a) const {
    const XYZ &b = body[a];
    return XYZ(com.x + Q[0][0] * b.x + Q[0][1] * b.y + Q[0][2] * b.z,
               com.y + Q[1][0] * b.x + Q[1][1] * b.y + Q[1][2] * b.z,
               com.z + Q[2][0] * b.x + Q[2][1] * b.y + Q[2][2] * b.z);
  }

  //! Space-frame angular momentum, Q pi
  XYZ SpaceAngularMomentum() const { return ToSpace(pi); }

  XYZ ToSpace(const XYZ &v) const {
    return XYZ(Q[0][0] * v.x + Q[0][1] * v.y + Q[0][2] * v.z,
               Q[1][0] * v.x + Q[1][1] * v.y + Q[1][2] * v.z,
               Q[2][0] * v.x + Q[2][1] * v.y + Q[2][2] * v.z);
  }

  XYZ ToBody(const XYZ &v) const {
    return XYZ(Q[0][0] * v.x + Q[1][0] * v.y + Q[2][0] * v.z,
               Q[0][1] * v.x + Q[1][1] * v.y + Q[2][1] * v.z,
               Q[0][2] * v.x + Q[1][2] * v.y + Q[2][2] * v.z);
  }

  double Kinetic() const {
    double k = 0.5 * mom.LengthSq() / mass;
    const double p[3] = {pi.x, pi.y, pi.z};
    for (uint i = 0; i < 3; ++i)
      if (active[i])
        k += 0.5 * p[i] * p[i] / I[i];
    return k;
  }

  //! Build the body frame from atom positions. The positions must be whole
  //! (unwrapped around the molecule), as given by BoxDimensions::UnwrapPBC.
  //! Momenta are zeroed.
  void Setup(const XYZ *pos, const double *m, const uint n);

  //! Maxwell-Boltzmann momenta at temperature kT (in K). normal() returns
  //! a standard normal deviate.
  template <class Normal> void DrawMomenta(const double kT, Normal &normal);

  //! Momentum update from the total force and the torque about the centre
  //! of mass, both in the space frame
  void Kick(const XYZ &force, const XYZ &torque, const double h);

  //! Free flight: centre-of-mass drift and the free-rotor splitting
  void Drift(const double h);

  //! Reverse the momenta (the involution that makes the HMC proposal exact)
  void FlipMomenta() {
    mom = -mom;
    pi = -pi;
  }

private:
  void Rotate(const uint k, const double angle);
  void RotateInSpace(const XYZ &axis, const double angle);
};

namespace detail {

// Eigen-decomposition of a symmetric 3x3 matrix by cyclic Jacobi rotations.
// On return a holds the eigenvalues on its diagonal and v the eigenvectors in
// its columns.
inline void Jacobi3(double a[3][3], double v[3][3]) {
  for (uint i = 0; i < 3; ++i)
    for (uint j = 0; j < 3; ++j)
      v[i][j] = (i == j) ? 1.0 : 0.0;
  for (uint sweep = 0; sweep < 50; ++sweep) {
    const double off = a[0][1] * a[0][1] + a[0][2] * a[0][2] +
                       a[1][2] * a[1][2];
    const double scale =
        a[0][0] * a[0][0] + a[1][1] * a[1][1] + a[2][2] * a[2][2];
    if (off <= 1e-30 * scale || off == 0.0)
      return;
    for (uint p = 0; p < 2; ++p) {
      for (uint q = p + 1; q < 3; ++q) {
        if (a[p][q] == 0.0)
          continue;
        const double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                         (std::abs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
        for (uint k = 0; k < 3; ++k) {
          const double akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (uint k = 0; k < 3; ++k) {
          const double apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (uint k = 0; k < 3; ++k) {
          const double vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
}

} // namespace detail

inline void Body::Setup(const XYZ *pos, const double *m, const uint n) {
  mass = 0.0;
  XYZ c;
  atomMass.assign(m, m + n);
  for (uint a = 0; a < n; ++a) {
    mass += m[a];
    c += pos[a] * m[a];
  }
  com = c * (1.0 / mass);

  double t[3][3] = {{0.0}};
  for (uint a = 0; a < n; ++a) {
    const XYZ d = pos[a] - com;
    const double r2 = d.LengthSq();
    const double dv[3] = {d.x, d.y, d.z};
    for (uint i = 0; i < 3; ++i)
      for (uint j = 0; j < 3; ++j)
        t[i][j] += m[a] * ((i == j ? r2 : 0.0) - dv[i] * dv[j]);
  }
  detail::Jacobi3(t, Q);
  // keep Q a proper rotation
  const double det =
      Q[0][0] * (Q[1][1] * Q[2][2] - Q[1][2] * Q[2][1]) -
      Q[0][1] * (Q[1][0] * Q[2][2] - Q[1][2] * Q[2][0]) +
      Q[0][2] * (Q[1][0] * Q[2][1] - Q[1][1] * Q[2][0]);
  if (det < 0.0)
    for (uint i = 0; i < 3; ++i)
      Q[i][2] = -Q[i][2];

  double iMax = 0.0;
  for (uint k = 0; k < 3; ++k) {
    I[k] = t[k][k];
    iMax = std::max(iMax, I[k]);
  }
  // A moment negligible next to the largest belongs to an axis the atoms lie
  // on (linear molecule) or to a single atom; rotation about it moves
  // nothing, so it carries no degree of freedom.
  uint nActive = 0;
  for (uint k = 0; k < 3; ++k) {
    active[k] = I[k] > 1e-8 * iMax && I[k] > 0.0;
    nActive += active[k];
  }
  // The two moments of a linear molecule are equal in exact arithmetic; make
  // them equal in floating point too, since the exact free flow assumes it.
  linear = (nActive == 2);
  if (linear) {
    double sum = 0.0;
    for (uint k = 0; k < 3; ++k)
      if (active[k])
        sum += I[k];
    for (uint k = 0; k < 3; ++k)
      if (active[k])
        I[k] = 0.5 * sum;
  }

  body.resize(n);
  for (uint a = 0; a < n; ++a)
    body[a] = ToBody(pos[a] - com);
  mom = XYZ();
  pi = XYZ();
}

template <class Normal>
inline void Body::DrawMomenta(const double kT, Normal &normal) {
  const double sm = std::sqrt(mass * kT);
  mom = XYZ(sm * normal(), sm * normal(), sm * normal());
  double p[3];
  for (uint k = 0; k < 3; ++k)
    p[k] = active[k] ? std::sqrt(I[k] * kT) * normal() : 0.0;
  pi = XYZ(p[0], p[1], p[2]);
}

inline void Body::Kick(const XYZ &force, const XYZ &torque, const double h) {
  mom += force * h;
  const XYZ tb = ToBody(torque);
  const double t[3] = {tb.x, tb.y, tb.z};
  double p[3] = {pi.x, pi.y, pi.z};
  for (uint k = 0; k < 3; ++k)
    if (active[k])
      p[k] += h * t[k];
  pi = XYZ(p[0], p[1], p[2]);
}

// Exact flow of pi_k^2 / 2 I_k for time h is a rotation by angle
// h pi_k / I_k about body axis k. The body turns forward: Q <- Q R_k(angle).
// Space-frame angular momentum is untouched, so pi = Q^T L turns backward:
// pi <- R_k(-angle) pi, leaving pi_k unchanged.
inline void Body::Rotate(const uint k, const double angle) {
  const uint i = (k + 1) % 3, j = (k + 2) % 3;
  const double c = std::cos(angle), s = std::sin(angle);
  for (uint r = 0; r < 3; ++r) {
    const double qi = Q[r][i], qj = Q[r][j];
    Q[r][i] = c * qi + s * qj;
    Q[r][j] = -s * qi + c * qj;
  }
  double p[3] = {pi.x, pi.y, pi.z};
  const double pi_i = p[i], pi_j = p[j];
  p[i] = c * pi_i + s * pi_j;
  p[j] = -s * pi_i + c * pi_j;
  pi = XYZ(p[0], p[1], p[2]);
}

// Rotate the molecule about a space-frame unit axis: Q <- R(axis, angle) Q
inline void Body::RotateInSpace(const XYZ &n, const double angle) {
  const double c = std::cos(angle), s = std::sin(angle);
  for (uint col = 0; col < 3; ++col) {
    const XYZ v(Q[0][col], Q[1][col], Q[2][col]);
    const XYZ r = v * c + geom::Cross(n, v) * s + n * (geom::Dot(n, v) * (1.0 - c));
    Q[0][col] = r.x;
    Q[1][col] = r.y;
    Q[2][col] = r.z;
  }
}

inline void Body::Drift(const double h) {
  com += mom * (h / mass);
  if (linear) {
    // exact free flow: rotation about L, which stays perpendicular to the
    // molecular axis, so the axial component of pi stays zero
    const XYZ L = ToSpace(pi);
    const double len = L.Length();
    if (len == 0.0)
      return;
    const double iPerp = active[0] ? I[0] : I[1];
    RotateInSpace(L * (1.0 / len), h * len / iPerp);
    pi = ToBody(L);
    double p[3] = {pi.x, pi.y, pi.z};
    for (uint k = 0; k < 3; ++k)
      if (!active[k])
        p[k] = 0.0;
    pi = XYZ(p[0], p[1], p[2]);
    return;
  }
  // R1(h/2) R2(h/2) R3(h) R2(h/2) R1(h/2); an inactive axis is skipped, and
  // its pi component stays zero. Each angle is read just before its own
  // rotation, since the earlier rotations change the other components.
  const double half = 0.5 * h;
  const uint order[5] = {0, 1, 2, 1, 0};
  const double frac[5] = {half, half, h, half, half};
  for (uint s = 0; s < 5; ++s) {
    const uint k = order[s];
    if (!active[k])
      continue;
    const double pk = (k == 0) ? pi.x : (k == 1) ? pi.y : pi.z;
    Rotate(k, frac[s] * pk / I[k]);
  }
}

//! Total force and torque about the centre of mass from per-atom forces
inline void ForceAndTorque(const Body &b, const XYZ *atomForce, XYZ &force,
                           XYZ &torque) {
  force = XYZ();
  torque = XYZ();
  for (uint a = 0; a < b.NumAtoms(); ++a) {
    force += atomForce[a];
    torque += geom::Cross(b.ToSpace(b.body[a]), atomForce[a]);
  }
}

} // namespace rigid

#endif /*RIGIDBODY_H*/
