// Standalone cache-behaviour probe for the whole-box pair sweep.
//
// Isolates the question "does the j-stream fall out of L2, and where" from
// everything else a GEMC run does. It replays BoxDimensions::DistSqRange over
// packed arrays of varying N with the same two-pass structure as
// CalculateEnergy::BoxInterTemplate's whole-box path, and reports cycles per
// pair and achieved bandwidth.
//
// The prediction under test: cycles/pair is flat while N * ~30 B fits L2 and
// steps up once it does not. For a 1 MiB L2 that is N ~ 34,500 atoms.
//
// Build (match the simulation's flags):
//   icpx -O3 -march=znver5 -fopenmp -I src -I lib tools/sweep_bench.cpp \
//        -o sweep_bench
// Run:
//   OMP_NUM_THREADS=4 ./sweep_bench
//
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#define GOMC_DISTSQ_CHUNK 256

static inline double MinImageSigned(double v, double ax, double halfAx) {
  if (v > halfAx)
    v -= ax;
  else if (v < -halfAx)
    v += ax;
  return v;
}

// Identical body to BoxDimensions::DistSqRange.
static inline void DistSqRange(double *__restrict distSq, const double xi,
                               const double yi, const double zi,
                               const double *__restrict xj,
                               const double *__restrict yj,
                               const double *__restrict zj, const int m,
                               const double axX, const double axY,
                               const double axZ, const double hX,
                               const double hY, const double hZ) {
  for (int k = 0; k < m; ++k) {
    double dx = xi - xj[k];
    double dy = yi - yj[k];
    double dz = zi - zj[k];
    dx = MinImageSigned(dx, axX, hX);
    dy = MinImageSigned(dy, axY, hY);
    dz = MinImageSigned(dz, axZ, hZ);
    distSq[k] = dx * dx + dy * dy + dz * dz;
  }
}

static inline uint64_t rdtscp() {
  unsigned aux;
  return __builtin_ia32_rdtscp(&aux);
}

int main() {
  const double rho = 0.0334 * 4.0; // water sites per A^3
  const double rCut = 25.0;        // the 10k system's RcutCoulomb

  printf("%8s %8s %8s %9s %10s %12s %10s\n", "waters", "N", "L(A)", "sweepKiB",
         "cyc/pair", "pairs", "GB/s_eff");

  for (int waters : {1000, 2000, 4000, 6000, 8000, 10000, 12000, 16000, 24000}) {
    const int N = waters * 4;
    const double L = std::cbrt(N / rho);
    std::vector<double> X(N), Y(N), Z(N);
    std::vector<int> Mol(N);
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> U(0.0, L);
    for (int i = 0; i < N; ++i) {
      X[i] = U(rng); Y[i] = U(rng); Z[i] = U(rng); Mol[i] = i / 4;
    }
    const double *pX = X.data(), *pY = Y.data(), *pZ = Z.data();
    const int *pMol = Mol.data();
    const double h = L * 0.5, rCutSq = rCut * rCut;

    // One pass of the triangular sweep, same shape as the real loop.
    double sink = 0.0;
    long long npairs = 0;
    const uint64_t t0 = rdtscp();
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1) reduction(+ : sink, npairs)
#endif
    for (int i = 0; i < N; i++) {
      const int currMol = pMol[i];
      const double xi = pX[i], yi = pY[i], zi = pZ[i];
      double distSq[GOMC_DISTSQ_CHUNK];
      for (int base = i + 1; base < N; base += GOMC_DISTSQ_CHUNK) {
        const int m = std::min(GOMC_DISTSQ_CHUNK, N - base);
        DistSqRange(distSq, xi, yi, zi, pX + base, pY + base, pZ + base, m, L,
                    L, L, h, h, h);
        npairs += m;
        for (int k = 0; k < m; k++) {
          if (currMol == pMol[base + k]) continue;
          if (!(rCutSq > distSq[k])) continue;
          sink += distSq[k]; // stand-in for the pair kernel
        }
      }
    }
    const uint64_t cycles = rdtscp() - t0;
    const double sweepKiB = N * 30.0 / 1024.0;
    // Bytes the j-stream must supply: 28 B per candidate (24 coords + 4 mol).
    const double bytes = (double)npairs * 28.0;
    printf("%8d %8d %8.1f %9.0f %10.3f %12lld %10.1f   %s\n", waters, N, L,
           sweepKiB, (double)cycles / npairs, npairs,
           bytes / ((double)cycles / 5.0e9) / 1e9,
           sweepKiB * 1024 > 1024 * 1024 ? "> L2" : "");
    if (sink == 1.2345) printf(""); // keep the compiler honest
  }
  return 0;
}
