/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#ifndef CLUSTER_PAIR_KERNEL_H
#define CLUSTER_PAIR_KERNEL_H

//
// Full-box short-range energy and forces on a cluster-pair list: the tiled
// kernel of rigid_body_hmc_design.md, "Tiled kernel: design decisions".
//
// Atoms are grouped spatially into clusters of 4 (x-y columns, sorted in z,
// cut into fours). Every pair of an i-cluster against a j-cluster -- a 4x4
// tile -- is computed with dense SIMD arithmetic. Pairs outside the cutoff,
// pairs in the same molecule, padding atoms, and j <= i on a diagonal tile
// are masked rather than branched on or compacted away. The j-forces of a
// tile are summed in registers and written once, contiguously, per tile, so
// there is no per-pair scatter; forces live in cluster order until AddForces
// permutes them back.
//
// The list is a half list over cluster pairs (ci <= cj) with radius equal to
// the cell-list cutoff and no skin. It is rebuilt on every evaluation, so it
// is complete for the positions it is used with and the forces are an exact
// function of the coordinates.
//
// The kernel evaluates what BoxForceTemplate<false, HasCharge, BoxDimensions,
// FF> evaluates, for an orthogonal box, no fractional molecule, a Mie vdW
// with one fixed exponent (12 or 16) and plain or shifted truncation, and the
// tabulated real-space Ewald term. The caller checks those conditions and
// keeps the reference kernel for everything else. Compute returns false when
// a Coulomb pair falls outside the table; the caller then uses the reference
// kernel for this evaluation.
//

#include <algorithm>
#include <cmath>
#include <vector>
#if defined _OPENMP
#include <omp.h>
#endif

#include "BasicTypes.h"
#include "EwaldRealTable.h"
#include "SimdLayer.h"
#include "VdwEvaluators.h"

class ClusterPairKernel {
public:
  static constexpr int CS = 4; //!< atoms per cluster

  struct Params {
    double L[3];
    double rCutSqBox, rCutSqVdw, rCutSqCoul;
    double tabLo, tabHi;
    EwaldRealTable::View tab;
    double qqFact;
    ff::VdwParams vp;
    uint stride; //!< kind count, for the flat pair index
  };

  //
  // Clusters and the pair list, from the per-atom arrays in cell order
  // (CalculateEnergy::BuildCellOrdered). atom[k] is the particle index of
  // slot k. rList is the list radius: the largest cutoff.
  //
  void Build(const int n, const double *x, const double *y, const double *z,
             const double *q, const int *mol, const int *kind,
             const int *atom, const double L[3], const double rList);

  //
  // Energies and cluster-order forces. Returns false, with energies and
  // forces unusable, when a Coulomb pair inside the cutoff lies outside the
  // real-space table.
  //
  template <class B, bool HasCharge, bool Shift, uint NEXP>
  bool Compute(const Params &P, double &enReal, double &enLJ);

  //! Add the forces of the last successful Compute to atom-order arrays.
  void AddForces(double *ax, double *ay, double *az) const {
    const int ns = CS * nClusters;
    for (int s = 0; s < ns; ++s) {
      const int a = atomC[s];
      if (a >= 0) {
        ax[a] += fX[s];
        ay[a] += fY[s];
        az[a] += fZ[s];
      }
    }
  }

  //! For CalculateEnergy::CheckClusterForce.
  int Clusters() const { return nClusters; }
  long PairListSize() const { return listSize; }
  const std::vector<int> &SlotAtoms() const { return atomC; }
  const std::vector<double> &ForceX() const { return fX; }
  const std::vector<double> &ForceY() const { return fY; }
  const std::vector<double> &ForceZ() const { return fZ; }

private:
  bool Near(const int a, const int b) const {
    double d2 = 0.0;
    for (int ax = 0; ax < 3; ++ax) {
      double d = bbCen[ax][b] - bbCen[ax][a];
      d = std::fabs(d - boxL[ax] * std::rint(d * invL[ax]));
      const double gap = d - (bbHalf[ax][a] + bbHalf[ax][b]);
      if (gap > 0.0)
        d2 += gap * gap;
    }
    return d2 < rListSq;
  }
  bool ColumnsNear(const int ca, const int cb) const {
    int dx = std::abs(colX[ca] - colX[cb]);
    int dy = std::abs(colY[ca] - colY[cb]);
    dx = std::min(dx, nx - dx);
    dy = std::min(dy, ny - dy);
    const double gx = std::max(0, dx - 1) * wx;
    const double gy = std::max(0, dy - 1) * wy;
    return gx * gx + gy * gy < rListSq;
  }
  void ScanPairs(const int ci, std::vector<int> &out) const;

  // Grid.
  int nx = 0, ny = 0, nCols = 0;
  double wx = 0.0, wy = 0.0;
  double boxL[3] = {0.0, 0.0, 0.0}, invL[3] = {0.0, 0.0, 0.0};
  double rListSq = 0.0;
  std::vector<int> colX, colY, colStart; // colStart: first cluster per column
  std::vector<int> colCount, order, colOfAtom;
  std::vector<double> zw;
  // Columns near column c, ascending: nbrCols[c * maxNbr + k], k < nbrLen[c].
  int maxNbr = 0;
  std::vector<int> nbrCols, nbrLen;

  // Clusters, slot-major: slot 4*c + u.
  int nClusters = 0;
  std::vector<double> xC, yC, zC, qC, molC;
  std::vector<int> kindC, atomC, clusterCol;
  std::vector<double> bbCen[3], bbHalf[3]; // per axis, so ScanPairs can load W

  // Half list: the j-clusters of ci are jPtr[ci][0 .. jLen[ci]), held in
  // per-thread buffers.
  std::vector<std::vector<int>> tBuf;
  std::vector<int> jThread, jOff, jLen;
  std::vector<const int *> jPtr;
  long listSize = 0;

  // Cluster-order forces.
  std::vector<double> fX, fY, fZ;
};

inline void ClusterPairKernel::Build(const int n, const double *x,
                                     const double *y, const double *z,
                                     const double *q, const int *mol,
                                     const int *kind, const int *atom,
                                     const double L[3], const double rList) {
  for (int ax = 0; ax < 3; ++ax) {
    boxL[ax] = L[ax];
    invL[ax] = 1.0 / L[ax];
  }
  // The margin covers rounding between the wrapped coordinates the bounding
  // boxes use and the unwrapped ones the kernel's minimum image uses.
  rListSq = (rList + 1.0e-6) * (rList + 1.0e-6);

  // Columns sized so a cluster of 4 is roughly as tall as it is wide.
  const double s = std::cbrt(CS * L[0] * L[1] * L[2] / std::max(n, 1));
  nx = std::max(1, (int)(L[0] / s));
  ny = std::max(1, (int)(L[1] / s));
  nCols = nx * ny;
  wx = L[0] / nx;
  wy = L[1] / ny;

  // Neighbour columns: the x and y column offsets that can be within rList,
  // all of them when the window covers the whole row.
  const int kx = (int)(std::sqrt(rListSq) / wx) + 1;
  const int ky = (int)(std::sqrt(rListSq) / wy) + 1;
  const int spanX = std::min(nx, 2 * kx + 1), spanY = std::min(ny, 2 * ky + 1);
  maxNbr = spanX * spanY;

  colCount.assign(nCols + 1, 0);
  order.resize(n);
  zw.resize(n);
  colOfAtom.resize(n);
  colStart.resize(nCols + 1);
  colX.resize(nCols);
  colY.resize(nCols);
  nbrCols.resize((size_t)nCols * maxNbr);
  nbrLen.resize(nCols);
#if defined _OPENMP
#pragma omp parallel
#endif
  {
    // Bin by wrapped x-y.
#if defined _OPENMP
#pragma omp for schedule(static)
#endif
    for (int a = 0; a < n; ++a) {
      const double xw = x[a] - L[0] * std::floor(x[a] * invL[0]);
      const double yw = y[a] - L[1] * std::floor(y[a] * invL[1]);
      zw[a] = z[a] - L[2] * std::floor(z[a] * invL[2]);
      const int cx = std::min(nx - 1, std::max(0, (int)(xw / wx)));
      const int cy = std::min(ny - 1, std::max(0, (int)(yw / wy)));
      colOfAtom[a] = cx + nx * cy;
    }

    // Counting sort into columns, and the cluster count of each column.
#if defined _OPENMP
#pragma omp single
#endif
    {
      for (int a = 0; a < n; ++a)
        ++colCount[colOfAtom[a] + 1];
      for (int c = 0; c < nCols; ++c)
        colCount[c + 1] += colCount[c];
      std::vector<int> fill(colCount.begin(), colCount.end() - 1);
      for (int a = 0; a < n; ++a)
        order[fill[colOfAtom[a]]++] = a;
      nClusters = 0;
      for (int c = 0; c < nCols; ++c) {
        colStart[c] = nClusters;
        colX[c] = c % nx;
        colY[c] = c / nx;
        nClusters += (colCount[c + 1] - colCount[c] + CS - 1) / CS;
      }
      colStart[nCols] = nClusters;
      const int ns = CS * nClusters;
      xC.resize(ns);
      yC.resize(ns);
      zC.resize(ns);
      qC.resize(ns);
      molC.resize(ns);
      kindC.resize(ns);
      atomC.resize(ns);
      clusterCol.resize(nClusters);
      for (int ax = 0; ax < 3; ++ax) {
        bbCen[ax].resize(nClusters);
        bbHalf[ax].resize(nClusters);
      }
    }

    // Sort each column by wrapped z and cut it into clusters, padding the
    // last cluster of the column.
#if defined _OPENMP
#pragma omp for schedule(dynamic, 8)
#endif
    for (int c = 0; c < nCols; ++c) {
      std::sort(order.begin() + colCount[c], order.begin() + colCount[c + 1],
                [&](const int a, const int b) { return zw[a] < zw[b]; });
      for (int ci = colStart[c]; ci < colStart[c + 1]; ++ci) {
        clusterCol[ci] = c;
        const int first = colCount[c] + CS * (ci - colStart[c]);
        const int last = std::min(first + CS, colCount[c + 1]);
        double lo[3] = {1e300, 1e300, 1e300},
               hi[3] = {-1e300, -1e300, -1e300};
        for (int u = 0; u < CS; ++u) {
          const int s4 = CS * ci + u;
          const bool real = first + u < last;
          // Padding copies the cluster's first atom's position, so every
          // distance stays finite; it is masked out by molC = -1.
          const int a = order[real ? first + u : first];
          xC[s4] = x[a];
          yC[s4] = y[a];
          zC[s4] = z[a];
          qC[s4] = real ? q[a] : 0.0;
          molC[s4] = real ? (double)mol[a] : -1.0;
          kindC[s4] = real ? kind[a] : 0;
          atomC[s4] = real ? atom[a] : -1;
          if (real) {
            const double w[3] = {x[a] - L[0] * std::floor(x[a] * invL[0]),
                                 y[a] - L[1] * std::floor(y[a] * invL[1]),
                                 zw[a]};
            for (int ax = 0; ax < 3; ++ax) {
              lo[ax] = std::min(lo[ax], w[ax]);
              hi[ax] = std::max(hi[ax], w[ax]);
            }
          }
        }
        for (int ax = 0; ax < 3; ++ax) {
          bbCen[ax][ci] = 0.5 * (lo[ax] + hi[ax]);
          bbHalf[ax][ci] = 0.5 * (hi[ax] - lo[ax]);
        }
      }
    }

    // Columns near each column, ascending so the j-lists keep the order of
    // a scan over all columns.
#if defined _OPENMP
#pragma omp for schedule(static)
#endif
    for (int c = 0; c < nCols; ++c) {
      auto within = [](const int d, const int n, const int k) {
        const int ad = std::abs(d);
        return std::min(ad, n - ad) <= k;
      };
      int *const out = nbrCols.data() + (size_t)c * maxNbr;
      int len = 0;
      for (int cy = 0; cy < ny; ++cy) {
        if (!within(cy - colY[c], ny, ky))
          continue;
        for (int cx = 0; cx < nx; ++cx) {
          const int cn = cx + nx * cy;
          if (within(cx - colX[c], nx, kx) && ColumnsNear(c, cn))
            out[len++] = cn;
        }
      }
      nbrLen[c] = len;
    }
  }

  const int ns = CS * nClusters;

  // Pair list, one pass. Each thread appends to its own buffer, which only
  // grows, and records where each of its i-clusters' j-lists starts; the
  // pointers are resolved once every buffer has stopped growing.
#if defined _OPENMP
  const int nThreads = omp_get_max_threads();
#else
  const int nThreads = 1;
#endif
  if ((int)tBuf.size() < nThreads)
    tBuf.resize(nThreads);
  jThread.resize(nClusters);
  jOff.resize(nClusters);
  jLen.resize(nClusters);
  jPtr.resize(nClusters);
#if defined _OPENMP
#pragma omp parallel
#endif
  {
#if defined _OPENMP
    const int t = omp_get_thread_num();
#else
    const int t = 0;
#endif
    std::vector<int> &buf = tBuf[t];
    buf.clear();
#if defined _OPENMP
#pragma omp for schedule(dynamic, 16)
#endif
    for (int ci = 0; ci < nClusters; ++ci) {
      jThread[ci] = t;
      jOff[ci] = (int)buf.size();
      ScanPairs(ci, buf);
      jLen[ci] = (int)buf.size() - jOff[ci];
    }
  }
  listSize = 0;
  for (int ci = 0; ci < nClusters; ++ci) {
    jPtr[ci] = tBuf[jThread[ci]].data() + jOff[ci];
    listSize += jLen[ci];
  }

  fX.resize(ns);
  fY.resize(ns);
  fZ.resize(ns);
}

inline void ClusterPairKernel::ScanPairs(const int ci,
                                         std::vector<int> &out) const {
  // The bounding-box test of Near, W j-clusters at a time; the clusters of a
  // column are contiguous.
  typedef simd::Native B;
  typedef B::Vec V;
  constexpr int W = B::W;
  const V zero = B::Set1(0.0), lim = B::Set1(rListSq);
  V cI[3], hI[3], Lv[3], iLv[3];
  for (int ax = 0; ax < 3; ++ax) {
    cI[ax] = B::Set1(bbCen[ax][ci]);
    hI[ax] = B::Set1(bbHalf[ax][ci]);
    Lv[ax] = B::Set1(boxL[ax]);
    iLv[ax] = B::Set1(invL[ax]);
  }
  const int coli = clusterCol[ci];
  const int *const nb = nbrCols.data() + (size_t)coli * maxNbr;
  for (int k = 0; k < nbrLen[coli]; ++k) {
    const int c = nb[k];
    const int end = colStart[c + 1];
    int cj = std::max(colStart[c], ci);
    for (; cj + W <= end; cj += W) {
      V d2 = zero;
      for (int ax = 0; ax < 3; ++ax) {
        V d = B::Sub(B::Load(bbCen[ax].data() + cj), cI[ax]);
        d = B::Sub(d, B::Mul(Lv[ax], B::Rint(B::Mul(d, iLv[ax]))));
        d = B::Select(B::Lt(d, zero), B::Sub(zero, d), d);
        V gap = B::Sub(d, B::Add(hI[ax], B::Load(bbHalf[ax].data() + cj)));
        gap = B::Select(B::Lt(zero, gap), gap, zero);
        d2 = B::Add(d2, B::Mul(gap, gap));
      }
      for (unsigned bits = B::Bits(B::Lt(d2, lim)); bits; bits &= bits - 1)
        out.push_back(cj + __builtin_ctz(bits));
    }
    for (; cj < end; ++cj)
      if (Near(ci, cj))
        out.push_back(cj);
  }
}

template <class B, bool HasCharge, bool Shift, uint NEXP>
inline bool ClusterPairKernel::Compute(const Params &P, double &enReal,
                                       double &enLJ) {
  typedef typename B::Vec V;
  typedef typename B::Mask M;
  constexpr int W = B::W;
  constexpr int R = 16 / W; // registers per 4x4 tile
  // Lane l = 4 i + j; the diagonal tile keeps j > i only.
  alignas(64) static const double diag[16] = {0, 1, 1, 1, 0, 0, 1, 1,
                                              0, 0, 0, 1, 0, 0, 0, 0};

  const int ns = CS * nClusters;
  std::fill(fX.begin(), fX.end(), 0.0);
  std::fill(fY.begin(), fY.end(), 0.0);
  std::fill(fZ.begin(), fZ.end(), 0.0);
  double *const fx = fX.data();
  double *const fy = fY.data();
  double *const fz = fZ.data();
  const double *const xc = xC.data();
  const double *const yc = yC.data();
  const double *const zc = zC.data();
  const double *const qc = qC.data();
  const double *const mc = molC.data();
  const int *const kc = kindC.data();
  const int *const *const jp = jPtr.data();
  const int *const jn = jLen.data();
  const int nCl = nClusters;
  const double nOver6 = (double)NEXP / 6.0;

  double eR = 0.0, eL = 0.0;
  int bad = 0;
#if defined _OPENMP
#pragma omp parallel for schedule(static, 1)                                  \
    reduction(+ : eR, eL, fx[ : ns], fy[ : ns], fz[ : ns]) reduction(| : bad)
#endif
  for (int ci = 0; ci < nCl; ++ci) {
    const V zero = B::Set1(0.0), one = B::Set1(1.0);
    const V Lx = B::Set1(P.L[0]), Ly = B::Set1(P.L[1]), Lz = B::Set1(P.L[2]);
    const V iLx = B::Set1(1.0 / P.L[0]), iLy = B::Set1(1.0 / P.L[1]),
            iLz = B::Set1(1.0 / P.L[2]);
    const V rcBox = B::Set1(P.rCutSqBox), rcV = B::Set1(P.rCutSqVdw);
    const V rcC = B::Set1(P.rCutSqCoul);
    const V tLo = B::Set1(P.tabLo), tHi = B::Set1(P.tabHi);
    const V tInv = B::Set1(P.tab.invDx);
    const V vN6 = B::Set1(nOver6);

    // The i-cluster expanded to the 16-lane tile layout: lane l holds i-atom
    // l / 4.
    alignas(64) double xI[16], yI[16], zI[16], qI[16], mI[16];
    int kI[16];
    for (int l = 0; l < 16; ++l) {
      const int a = CS * ci + l / 4;
      xI[l] = xc[a];
      yI[l] = yc[a];
      zI[l] = zc[a];
      qI[l] = qc[a] * P.qqFact;
      mI[l] = mc[a];
      kI[l] = kc[a];
    }
    V fiX[R], fiY[R], fiZ[R];
    for (int r = 0; r < R; ++r)
      fiX[r] = fiY[r] = fiZ[r] = zero;
    V accR = zero, accL = zero;
    M badM = B::None();

    for (int jj = 0; jj < jn[ci]; ++jj) {
      const int cj = jp[ci][jj];
      const int b4 = CS * cj;
      const V xJ = B::LoadJ4(xc + b4), yJ = B::LoadJ4(yc + b4),
              zJ = B::LoadJ4(zc + b4), mJ = B::LoadJ4(mc + b4);
      V qJ = zero;
      if constexpr (HasCharge)
        qJ = B::LoadJ4(qc + b4);
      alignas(64) int idx[16];
      for (int l = 0; l < 16; ++l)
        idx[l] = kI[l] + kc[b4 + (l & 3)] * (int)P.stride;
      const M validJ = B::Ge(mJ, zero);
      V fjX = zero, fjY = zero, fjZ = zero;

      for (int r = 0; r < R; ++r) {
        V dx = B::Sub(B::Load(xI + r * W), xJ);
        V dy = B::Sub(B::Load(yI + r * W), yJ);
        V dz = B::Sub(B::Load(zI + r * W), zJ);
        dx = B::Sub(dx, B::Mul(Lx, B::Rint(B::Mul(dx, iLx))));
        dy = B::Sub(dy, B::Mul(Ly, B::Rint(B::Mul(dy, iLy))));
        dz = B::Sub(dz, B::Mul(Lz, B::Rint(B::Mul(dz, iLz))));
        const V d = B::Fma(dx, dx, B::Fma(dy, dy, B::Mul(dz, dz)));

        const V mi = B::Load(mI + r * W);
        M pair = B::And(B::Ne(mi, mJ), B::And(validJ, B::Ge(mi, zero)));
        if (ci == cj)
          pair = B::And(pair, B::Ne(B::Load(diag + r * W), zero));
        pair = B::And(pair, B::Lt(d, rcBox));

        // Lennard-Jones / Mie, `<=` as in FFAdapter's `rCut < distSq` test.
        const M mV = B::And(pair, B::Le(d, rcV));
        const V dV = B::Select(mV, d, one);
        const V inv = B::Div(one, dV);
        const int *ix = idx + r * W;
        const V rr = B::Mul(B::Gather(P.vp.sigmaSq, ix), inv);
        const V rr2 = B::Mul(rr, rr);
        const V att = B::Mul(rr2, rr);
        V rep;
        if constexpr (NEXP == 12) {
          rep = B::Mul(att, att);
        } else {
          static_assert(NEXP == 16, "fixed exponents are 12 and 16");
          const V rr4 = B::Mul(rr2, rr2);
          rep = B::Mul(rr4, rr4);
        }
        V enL = B::Mul(B::Gather(P.vp.epsilon_cn, ix), B::Sub(rep, att));
        if constexpr (Shift)
          enL = B::Sub(enL, B::Gather(P.vp.shiftConst, ix));
        const V virL =
            B::Mul(B::Mul(B::Gather(P.vp.epsilon_cn_6, ix),
                          B::Sub(B::Mul(vN6, rep), att)),
                   inv);
        accL = B::Add(accL, B::Select(mV, enL, zero));
        V vir = B::Select(mV, virL, zero);

        if constexpr (HasCharge) {
          const M mC = B::And(pair, B::Le(d, rcC));
          const M inTab = B::And(B::Ge(d, tLo), B::Lt(d, tHi));
          badM = B::Or(badM, B::AndNot(mC, inTab));
          // Every lane must index inside the table, masked ones included.
          const M okC = B::And(mC, inTab);
          const V t = B::Mul(B::Sub(B::Select(okC, d, tLo), tLo), tInv);
          V e, v;
          B::TableEnVir(P.tab.c, t, e, v);
          const V qq = B::Mul(B::Load(qI + r * W), qJ);
          accR = B::Add(accR, B::Select(okC, B::Mul(qq, e), zero));
          vir = B::Add(vir, B::Select(okC, B::Mul(qq, v), zero));
        }

        const V px = B::Mul(dx, vir), py = B::Mul(dy, vir),
                pz = B::Mul(dz, vir);
        fiX[r] = B::Add(fiX[r], px);
        fiY[r] = B::Add(fiY[r], py);
        fiZ[r] = B::Add(fiZ[r], pz);
        fjX = B::Add(fjX, px);
        fjY = B::Add(fjY, py);
        fjZ = B::Add(fjZ, pz);
      }
      // One contiguous update of the j-cluster per tile.
      alignas(32) double sj[3][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
      B::FoldJ4Add(sj[0], fjX);
      B::FoldJ4Add(sj[1], fjY);
      B::FoldJ4Add(sj[2], fjZ);
      for (int u = 0; u < CS; ++u) {
        fx[b4 + u] -= sj[0][u];
        fy[b4 + u] -= sj[1][u];
        fz[b4 + u] -= sj[2][u];
      }
    }

    alignas(64) double t[3][16];
    for (int r = 0; r < R; ++r) {
      B::Store(t[0] + r * W, fiX[r]);
      B::Store(t[1] + r * W, fiY[r]);
      B::Store(t[2] + r * W, fiZ[r]);
    }
    for (int u = 0; u < CS; ++u) {
      const int l = 4 * u;
      fx[CS * ci + u] += (t[0][l] + t[0][l + 1]) + (t[0][l + 2] + t[0][l + 3]);
      fy[CS * ci + u] += (t[1][l] + t[1][l + 1]) + (t[1][l + 2] + t[1][l + 3]);
      fz[CS * ci + u] += (t[2][l] + t[2][l + 1]) + (t[2][l + 2] + t[2][l + 3]);
    }
    eR += B::HSum(accR);
    eL += B::HSum(accL);
    if (B::Any(badM))
      bad = 1;
  }

  enReal = eR;
  enLJ = eL;
  return bad == 0;
}

#endif /*CLUSTER_PAIR_KERNEL_H*/
