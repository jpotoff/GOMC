// Step 0 probe: is a closed-form erfc(alpha*r)/r faster and/or more accurate
// than GOMC's tabulated real-space kernel once both are vectorized?
//
// Everything here evaluates the same function the table stores,
//     f(s) = erfc(alpha*sqrt(s)) / sqrt(s),    s = distSq
// so the candidates are directly comparable.
//
// Accuracy is measured against std::erfcl in long double and is meaningful on
// any machine.  Timing is only meaningful where the loops actually vectorise,
// i.e. an AVX-512 target -- check with
//     objdump -d erfc_bench | grep -c 'vgatherqpd\|vgatherdpd'
//
//   icpx -O3 -march=znver5 -fopenmp -o erfc_bench erfc_bench.cpp
//   g++  -O3 -march=native -fopenmp -o erfc_bench erfc_bench.cpp
//
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#if defined(__AVX512F__)
#include <immintrin.h>
#define HAVE_AVX512 1
#endif

// ---------------------------------------------------------------- reference
static long double RefF(double s, double alpha) {
  const long double r = sqrtl((long double)s);
  return erfcl((long double)alpha * r) / r;
}

// ------------------------------------------------- candidate 1: GOMC table
// Cubic Hermite in distSq, 4 coefficients per interval -> 4 gathers per lane.
// Mirrors EwaldRealTable::Grid exactly (same dx default, same Hermite form).
struct Table4 {
  std::vector<double> c; // 4 per interval
  const double *cp = nullptr;
  double lo, invDx;
  int n;

  static double Exact(double s, double a) { return std::erfc(a * std::sqrt(s)) / std::sqrt(s); }
  static double dExact(double s, double a) {
    const double r = std::sqrt(s);
    const double d = -2.0 * a * std::exp(-a * a * s) / std::sqrt(M_PI);
    return (d / r - std::erfc(a * r) / s) * (0.5 / r);
  }
  static void Hermite(double *o, double v0, double v1, double m0, double m1) {
    o[0] = v0; o[1] = m0;
    o[2] = -3.0 * v0 + 3.0 * v1 - 2.0 * m0 - m1;
    o[3] = 2.0 * v0 - 2.0 * v1 + m0 + m1;
  }
  void Build(double a, double sLo, double sHi, double dxWanted) {
    lo = sLo;
    n = (int)std::ceil((sHi - sLo) / dxWanted);
    if (n < 64) n = 64;
    const double dx = (sHi - sLo) / n;
    invDx = 1.0 / dx;
    c.resize(4 * n);
    for (int i = 0; i < n; ++i) {
      const double x0 = lo + i * dx;
      Hermite(&c[4 * i], Exact(x0, a), Exact(x0 + dx, a),
              dExact(x0, a) * dx, dExact(x0 + dx, a) * dx);
    }
    cp = c.data();
  }
  struct Hot {
    const double *__restrict c;
    double lo, invDx;
    inline double Eval(double s) const {
      const double t = (s - lo) * invDx;
      const int i = (int)t;
      const double f = t - i;
      return c[4 * i] +
             f * (c[4 * i + 1] + f * (c[4 * i + 2] + f * c[4 * i + 3]));
    }
  };
  Hot hot() const { return {cp, lo, invDx}; }
  inline double Eval(double s) const { return hot().Eval(s); }
  size_t Bytes() const { return c.size() * sizeof(double); }
};

// ------------------------------------- candidate 2: linear table, 2 gathers
// Tests the hypothesis that the table's cost is gather count, not arithmetic:
// same storage budget spent on more intervals with fewer coefficients.
struct Table2 {
  std::vector<double> c; // 2 per interval: value, slope
  const double *cp = nullptr;
  double lo, invDx;
  int n;
  void Build(double a, double sLo, double sHi, double dxWanted) {
    lo = sLo;
    n = (int)std::ceil((sHi - sLo) / dxWanted);
    if (n < 64) n = 64;
    const double dx = (sHi - sLo) / n;
    invDx = 1.0 / dx;
    c.resize(2 * n);
    for (int i = 0; i < n; ++i) {
      const double x0 = lo + i * dx;
      const double v0 = Table4::Exact(x0, a), v1 = Table4::Exact(x0 + dx, a);
      c[2 * i] = v0;
      c[2 * i + 1] = v1 - v0;
    }
    cp = c.data();
  }
  struct Hot {
    const double *__restrict c;
    double lo, invDx;
    inline double Eval(double s) const {
      const double t = (s - lo) * invDx;
      const int i = (int)t;
      const double f = t - i;
      return c[2 * i] + f * c[2 * i + 1];
    }
  };
  Hot hot() const { return {cp, lo, invDx}; }
  inline double Eval(double s) const { return hot().Eval(s); }
  size_t Bytes() const { return c.size() * sizeof(double); }
};

// ------------------------------------------ candidate 3: Abramowitz & Stegun
// 7.1.26, the usual Ewald choice (LAMMPS). Quoted 1.5e-7 ABSOLUTE in erfc.
struct AS71126 {
  double alpha, alphaSq;
  void Build(double a) { alpha = a; alphaSq = a * a; }
  inline double Eval(double s) const {
    const double r = std::sqrt(s);
    const double x = alpha * r;
    const double t = 1.0 / (1.0 + 0.3275911 * x);
    const double poly =
        t * (0.254829592 +
             t * (-0.284496736 +
                  t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    return poly * std::exp(-alphaSq * s) / r;
  }
  typedef AS71126 Hot;
  Hot hot() const { return *this; }
  size_t Bytes() const { return 0; }
};

// ------------------------------- candidate 4: Chebyshev fit of erfc(x)e^{x^2}
// x = alpha*r is bounded by sqrt(-ln(tolerance)) ~ 3.39 at the cutoff,
// INDEPENDENT of the cutoff (Forcefield.cpp:81). So one fit over a fixed,
// narrow range suffices -- no range splitting, hence no branches.
// g(x) = erfc(x)*exp(x^2) is smooth and O(1); coefficients are generated here
// in long double rather than copied from anywhere.
template <int DEG> struct ChebErfc {
  double cf[DEG + 1];
  double alpha, alphaSq, xhi;
  void Build(double a, double xHi) {
    alpha = a; alphaSq = a * a; xhi = xHi;
    const int M = 4 * (DEG + 1);
    std::vector<long double> g(M);
    for (int j = 0; j < M; ++j) {
      const long double u = cosl(M_PIl * (j + 0.5L) / M);         // [-1,1]
      const long double x = 0.5L * (u + 1.0L) * (long double)xHi; // [0,xHi]
      g[j] = erfcl(x) * expl(x * x);
    }
    for (int k = 0; k <= DEG; ++k) {
      long double acc = 0.0L;
      for (int j = 0; j < M; ++j)
        acc += g[j] * cosl(M_PIl * k * (j + 0.5L) / M);
      cf[k] = (double)(2.0L / M * acc);
    }
    cf[0] *= 0.5;
  }
  inline double Eval(double s) const {
    const double r = std::sqrt(s);
    const double x = alpha * r;
    const double u = 2.0 * x / xhi - 1.0; // map to [-1,1]
    const double u2 = 2.0 * u;
    double b0 = 0.0, b1 = 0.0;            // Clenshaw, fixed trip count
    for (int k = DEG; k >= 1; --k) {
      const double b = u2 * b0 - b1 + cf[k];
      b1 = b0; b0 = b;
    }
    const double g = u * b0 - b1 + cf[0];
    return g * std::exp(-alphaSq * s) / r;
  }
  typedef ChebErfc Hot;
  Hot hot() const { return *this; }
  size_t Bytes() const { return 0; }
};

// ------------------------------------------------- candidate 5: scalar libm
struct Libm {
  double alpha;
  void Build(double a) { alpha = a; }
  inline double Eval(double s) const {
    const double r = std::sqrt(s);
    return std::erfc(alpha * r) / r;
  }
  typedef Libm Hot;
  Hot hot() const { return *this; }
  size_t Bytes() const { return 0; }
};

// ------------------------------- candidate 1b: table4, explicit AVX-512 gather
// Branch-free on purpose. After the step-1 list split the Coulomb survivors are
// all below rCutCoulombSq, so the upper range test is redundant; the only real
// exception is distSq below the table floor, i.e. an overlap, which is rare
// enough to belong in a scalar fixup rather than a per-pair branch.
#ifdef HAVE_AVX512
static double TimeTable4Gather(const Table4 &t, const std::vector<double> &s,
                               int reps, double &sink) {
  using clk = std::chrono::steady_clock;
  const int n = (int)s.size();
  const double *__restrict p = s.data();
  const double *__restrict c = t.cp;
  const __m512d vLo = _mm512_set1_pd(t.lo);
  const __m512d vInv = _mm512_set1_pd(t.invDx);
  __m512d vacc = _mm512_setzero_pd();
  auto t0 = clk::now();
  for (int r = 0; r < reps; ++r) {
    for (int i = 0; i + 8 <= n; i += 8) {
      const __m512d sv = _mm512_loadu_pd(p + i);
      const __m512d tv = _mm512_mul_pd(_mm512_sub_pd(sv, vLo), vInv);
      const __m256i iv = _mm512_cvttpd_epi32(tv);
      const __m512d fv = _mm512_sub_pd(tv, _mm512_cvtepi32_pd(iv));
      const __m256i base = _mm256_slli_epi32(iv, 2); // 4 doubles per interval
      const __m512d c0 = _mm512_i32gather_pd(base, c, 8);
      const __m512d c1 = _mm512_i32gather_pd(base, c + 1, 8);
      const __m512d c2 = _mm512_i32gather_pd(base, c + 2, 8);
      const __m512d c3 = _mm512_i32gather_pd(base, c + 3, 8);
      __m512d v = _mm512_fmadd_pd(fv, c3, c2);
      v = _mm512_fmadd_pd(fv, v, c1);
      v = _mm512_fmadd_pd(fv, v, c0);
      vacc = _mm512_add_pd(vacc, v);
    }
  }
  auto t1 = clk::now();
  sink += _mm512_reduce_add_pd(vacc);
  const double secs = std::chrono::duration<double>(t1 - t0).count();
  return secs * 1e9 / ((double)(n / 8 * 8) * reps);
}
#endif

// ----------------------------------------------------------------- harness
template <class K>
static void Accuracy(const char *name, const K &k, const std::vector<double> &s,
                     double alpha, double &maxRel, double &maxAbs,
                     double &atR) {
  maxRel = 0.0; maxAbs = 0.0; atR = 0.0;
  for (double si : s) {
    const long double ref = RefF(si, alpha);
    const long double got = (long double)k.Eval(si);
    const long double ae = fabsl(got - ref);
    if ((double)ae > maxAbs) maxAbs = (double)ae;
    if (ref != 0.0L) {
      const double re = (double)(ae / fabsl(ref));
      if (re > maxRel) { maxRel = re; atR = std::sqrt(si); }
    }
  }
}

template <class K>
static double Time(const K &k, const std::vector<double> &s, int reps,
                   double &sink) {
  using clk = std::chrono::steady_clock;
  const int n = (int)s.size();
  const double *__restrict p = s.data();
  const typename K::Hot h = k.hot(); // by value: cannot alias p
  double acc = 0.0;
  auto t0 = clk::now();
  for (int r = 0; r < reps; ++r) {
#pragma omp simd reduction(+ : acc)
    for (int i = 0; i < n; ++i)
      acc += h.Eval(p[i]);
  }
  auto t1 = clk::now();
  sink += acc;
  const double secs = std::chrono::duration<double>(t1 - t0).count();
  return secs * 1e9 / ((double)n * reps); // ns per evaluation
}

int main(int argc, char **argv) {
  // Defaults match test/input/Systems/OPC/GEMC/bench: Rcut 8, RcutCoulomb 12,
  // RcutLow 0.7, Tolerance 1e-5.
  const double rCutCoul = argc > 1 ? atof(argv[1]) : 12.0;
  const double tol      = argc > 2 ? atof(argv[2]) : 1e-5;
  const double rCutLow  = argc > 3 ? atof(argv[3]) : 0.7;

  const double alpha = std::sqrt(-std::log(tol)) / rCutCoul;
  const double sLo = std::max(rCutLow * rCutLow, 0.25); // EwaldRealTable floor
  const double sHi = rCutCoul * rCutCoul;
  const double xHi = alpha * rCutCoul;

  printf("rCutCoulomb %.1f  tolerance %.0e  ->  alpha %.6f   x in [%.3f, %.3f]\n",
         rCutCoul, tol, alpha, alpha * std::sqrt(sLo), xHi);
  printf("distSq in [%.3f, %.1f]\n\n", sLo, sHi);

  // Accuracy sweep: dense and deterministic, so the worst case is not missed.
  std::vector<double> sweep;
  const int NS = 2000000;
  sweep.reserve(NS);
  for (int i = 0; i < NS; ++i)
    sweep.push_back(sLo + (sHi - sLo) * (i + 0.5) / NS);

  // Timing set: r^2 sampled from a uniform-density shell, i.e. pdf(r) ~ r^2,
  // which is what the pair loop actually sees.
  std::vector<double> timing;
  const int NT = 1 << 20;
  timing.reserve(NT);
  {
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const double rLo = std::sqrt(sLo);
    const double lo3 = rLo * rLo * rLo, hi3 = rCutCoul * rCutCoul * rCutCoul;
    for (int i = 0; i < NT; ++i) {
      const double r = std::cbrt(lo3 + (hi3 - lo3) * u(rng));
      timing.push_back(r * r);
    }
  }

  Table4 t4;  t4.Build(alpha, sLo, sHi, 0.05);          // today's table
  Table2 t2;  t2.Build(alpha, sLo, sHi, 0.05 * 0.25);   // 4x finer, half the gathers
  AS71126 as; as.Build(alpha);
  ChebErfc<10> c10; c10.Build(alpha, xHi);
  ChebErfc<14> c14; c14.Build(alpha, xHi);
  ChebErfc<18> c18; c18.Build(alpha, xHi);
  Libm lm;    lm.Build(alpha);

  printf("%-16s %12s %12s %8s %8s %9s\n", "kernel", "max rel err",
         "max abs err", "at r", "KiB", "ns/eval");
  printf("%-16s %12s %12s %8s %8s %9s\n", "----------------", "------------",
         "------------", "--------", "--------", "---------");

  double sink = 0.0;
  const int reps = 40;
  auto row = [&](const char *name, auto &k) {
    double rel, abs_, atR;
    Accuracy(name, k, sweep, alpha, rel, abs_, atR);
    const double ns = Time(k, timing, reps, sink);
    printf("%-16s %12.3e %12.3e %8.2f %8.1f %9.3f\n", name, rel, abs_, atR,
           k.Bytes() / 1024.0, ns);
  };

  row("table4 (now)", t4);
  row("table2 linear", t2);
  row("A&S 7.1.26", as);
  row("cheb deg10", c10);
  row("cheb deg14", c14);
  row("cheb deg18", c18);
  row("libm erfc", lm);

#ifdef HAVE_AVX512
  {
    double rel, abs_, atR;
    Accuracy("x", t4, sweep, alpha, rel, abs_, atR);
    const double ns = TimeTable4Gather(t4, timing, reps, sink);
    printf("%-16s %12.3e %12.3e %8.2f %8.1f %9.3f\n", "table4 gather", rel,
           abs_, atR, t4.Bytes() / 1024.0, ns);
  }
#else
  printf("\n(no AVX-512 here: the explicit-gather table row is skipped, and\n"
         " every row above is at this host's vector width, not the target's)\n");
#endif

  printf("\n(sink %.6e)\n", sink);
  return 0;
}
