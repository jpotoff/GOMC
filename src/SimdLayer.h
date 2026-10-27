/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#ifndef SIMD_LAYER_H
#define SIMD_LAYER_H

//
// The few dozen double-precision vector operations the cluster-pair kernel
// (ClusterPairKernel.h) is written against.
//
// Three backends with the same interface:
//
//   Generic  4 lanes held in plain arrays. Runs on any CPU, which is what lets
//            the tile logic be tested on a machine without AVX.
//   Avx2     4 lanes, AVX2 + FMA (x86-64-v3).
//   Avx512   8 lanes, AVX-512F.
//
// `Native` is the widest one the compiler was told it may use. Defining
// GOMC_SIMD_GENERIC forces Generic, for comparing backends on one machine.
//
// Lane counts are 4 or 8 and the kernel's tile is 16 pairs (4 i-atoms by 4
// j-atoms), so a tile is 4 or 2 registers. The lane layout within a tile is
// i-major: pair (i, j) is lane 4*i + j. Hence LoadJ4, which repeats a
// j-cluster's 4 values across the register; LoadRows, which fills each group
// of 4 lanes (one i) from its own 4 contiguous values, row[0] for the first
// i of the register, row[1] for the second (W = 8 only); and FoldJ4, which
// sums the lanes that belong to the same j.
//
// The one non-elementwise operation that matters for speed is TableEnVir.
// Each pair's real-space Ewald coefficients (energy and virial cubics, see
// EwaldRealTable::EvalBothUnchecked) sit in one 64-byte line. Gathering them
// one coefficient at a time cost about a quarter of the per-atom vectorised
// kernel; here each lane's line is loaded whole and the lines are transposed
// in registers.
//

#include <cmath>
#include <cstdint>

#if !defined(GOMC_SIMD_GENERIC) &&                                            \
    (defined(__AVX512F__) || (defined(__AVX2__) && defined(__FMA__)))
#include <immintrin.h>
#endif

namespace simd {

// ------------------------------------------------------------- generic ----

struct Generic {
  static constexpr int W = 4;
  static const char *Name() { return "generic"; }
  struct Vec {
    double v[4];
  };
  struct Mask {
    bool m[4];
  };

  static Vec Set1(const double a) { return {{a, a, a, a}}; }
  static Vec Load(const double *p) { return {{p[0], p[1], p[2], p[3]}}; }
  static Vec LoadJ4(const double *p) { return Load(p); }
  static Vec LoadRows(const double *const *row) { return Load(row[0]); }
  static void Store(double *p, const Vec &a) {
    for (int k = 0; k < 4; ++k)
      p[k] = a.v[k];
  }
#define GOMC_SIMD_G_BINOP(NAME, EXPR)                                          \
  static Vec NAME(const Vec &a, const Vec &b) {                                \
    Vec r;                                                                     \
    for (int k = 0; k < 4; ++k)                                                \
      r.v[k] = EXPR;                                                           \
    return r;                                                                  \
  }
  GOMC_SIMD_G_BINOP(Add, a.v[k] + b.v[k])
  GOMC_SIMD_G_BINOP(Sub, a.v[k] - b.v[k])
  GOMC_SIMD_G_BINOP(Mul, a.v[k] * b.v[k])
  GOMC_SIMD_G_BINOP(Div, a.v[k] / b.v[k])
#undef GOMC_SIMD_G_BINOP
  //! a * b + c. Not std::fma: without FMA hardware that is a library call.
  static Vec Fma(const Vec &a, const Vec &b, const Vec &c) {
    Vec r;
    for (int k = 0; k < 4; ++k)
      r.v[k] = a.v[k] * b.v[k] + c.v[k];
    return r;
  }
  //! Round to nearest, ties to even: the rint in MinImageSigned.
  static Vec Rint(const Vec &a) {
    Vec r;
    for (int k = 0; k < 4; ++k)
      r.v[k] = std::rint(a.v[k]);
    return r;
  }
#define GOMC_SIMD_G_CMP(NAME, OP)                                              \
  static Mask NAME(const Vec &a, const Vec &b) {                               \
    Mask r;                                                                    \
    for (int k = 0; k < 4; ++k)                                                \
      r.m[k] = a.v[k] OP b.v[k];                                               \
    return r;                                                                  \
  }
  GOMC_SIMD_G_CMP(Lt, <)
  GOMC_SIMD_G_CMP(Le, <=)
  GOMC_SIMD_G_CMP(Ge, >=)
  GOMC_SIMD_G_CMP(Ne, !=)
#undef GOMC_SIMD_G_CMP
  static Mask And(const Mask &a, const Mask &b) {
    Mask r;
    for (int k = 0; k < 4; ++k)
      r.m[k] = a.m[k] && b.m[k];
    return r;
  }
  static Mask AndNot(const Mask &a, const Mask &b) { //!< a & ~b
    Mask r;
    for (int k = 0; k < 4; ++k)
      r.m[k] = a.m[k] && !b.m[k];
    return r;
  }
  static Mask Or(const Mask &a, const Mask &b) {
    Mask r;
    for (int k = 0; k < 4; ++k)
      r.m[k] = a.m[k] || b.m[k];
    return r;
  }
  static Mask None() { return {{false, false, false, false}}; }
  static bool Any(const Mask &a) {
    return a.m[0] || a.m[1] || a.m[2] || a.m[3];
  }
  //! m ? a : b
  static Vec Select(const Mask &m, const Vec &a, const Vec &b) {
    Vec r;
    for (int k = 0; k < 4; ++k)
      r.v[k] = m.m[k] ? a.v[k] : b.v[k];
    return r;
  }
  static double HSum(const Vec &a) {
    return (a.v[0] + a.v[1]) + (a.v[2] + a.v[3]);
  }
  //! out[j] += sum of the lanes holding j-atom j.
  static void FoldJ4Add(double *out, const Vec &a) {
    for (int k = 0; k < 4; ++k)
      out[k] += a.v[k];
  }
  //! base[idx[k]] per lane.
  static Vec Gather(const double *base, const int *idx) {
    return {{base[idx[0]], base[idx[1]], base[idx[2]], base[idx[3]]}};
  }
  //
  // Energy and virial cubics of the real-space table at t = (distSq - lo) *
  // invDx, which every lane must hold inside the table. Same arithmetic as
  // EwaldRealTable::EvalBothUnchecked.
  //
  static void TableEnVir(const double *c, const Vec &t, Vec &en, Vec &vir) {
    for (int k = 0; k < 4; ++k) {
      const int i = (int)t.v[k];
      const double f = t.v[k] - i;
      const double *p = c + 8 * i;
      en.v[k] = p[0] + f * (p[1] + f * (p[2] + f * p[3]));
      vir.v[k] = p[4] + f * (p[5] + f * (p[6] + f * p[7]));
    }
  }
};

// ---------------------------------------------------------------- AVX2 ----

#if !defined(GOMC_SIMD_GENERIC) && defined(__AVX2__) && defined(__FMA__)
struct Avx2 {
  static constexpr int W = 4;
  static const char *Name() { return "AVX2"; }
  typedef __m256d Vec;
  typedef __m256d Mask;

  static Vec Set1(const double a) { return _mm256_set1_pd(a); }
  static Vec Load(const double *p) { return _mm256_loadu_pd(p); }
  static Vec LoadJ4(const double *p) { return _mm256_loadu_pd(p); }
  static Vec LoadRows(const double *const *row) {
    return _mm256_loadu_pd(row[0]);
  }
  static void Store(double *p, const Vec a) { _mm256_storeu_pd(p, a); }
  static Vec Add(const Vec a, const Vec b) { return _mm256_add_pd(a, b); }
  static Vec Sub(const Vec a, const Vec b) { return _mm256_sub_pd(a, b); }
  static Vec Mul(const Vec a, const Vec b) { return _mm256_mul_pd(a, b); }
  static Vec Div(const Vec a, const Vec b) { return _mm256_div_pd(a, b); }
  static Vec Fma(const Vec a, const Vec b, const Vec c) {
    return _mm256_fmadd_pd(a, b, c);
  }
  static Vec Rint(const Vec a) {
    return _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  }
  static Mask Lt(const Vec a, const Vec b) {
    return _mm256_cmp_pd(a, b, _CMP_LT_OQ);
  }
  static Mask Le(const Vec a, const Vec b) {
    return _mm256_cmp_pd(a, b, _CMP_LE_OQ);
  }
  static Mask Ge(const Vec a, const Vec b) {
    return _mm256_cmp_pd(a, b, _CMP_GE_OQ);
  }
  static Mask Ne(const Vec a, const Vec b) {
    return _mm256_cmp_pd(a, b, _CMP_NEQ_OQ);
  }
  static Mask And(const Mask a, const Mask b) { return _mm256_and_pd(a, b); }
  static Mask AndNot(const Mask a, const Mask b) {
    return _mm256_andnot_pd(b, a);
  }
  static Mask Or(const Mask a, const Mask b) { return _mm256_or_pd(a, b); }
  static Mask None() { return _mm256_setzero_pd(); }
  static bool Any(const Mask a) { return _mm256_movemask_pd(a) != 0; }
  static Vec Select(const Mask m, const Vec a, const Vec b) {
    return _mm256_blendv_pd(b, a, m);
  }
  static double HSum(const Vec a) {
    const __m128d lo = _mm256_castpd256_pd128(a);
    const __m128d hi = _mm256_extractf128_pd(a, 1);
    const __m128d s = _mm_add_pd(lo, hi);
    return _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s, s)));
  }
  static void FoldJ4Add(double *out, const Vec a) {
    _mm256_storeu_pd(out, _mm256_add_pd(_mm256_loadu_pd(out), a));
  }
  static Vec Gather(const double *base, const int *idx) {
    return _mm256_i32gather_pd(base, _mm_loadu_si128((const __m128i *)idx),
                               8);
  }
  static void TableEnVir(const double *c, const Vec t, Vec &en, Vec &vir) {
    const __m128i i4 = _mm256_cvttpd_epi32(t);
    const Vec f = _mm256_sub_pd(t, _mm256_cvtepi32_pd(i4));
    alignas(16) int ix[4];
    _mm_store_si128((__m128i *)ix, i4);
    const double *p0 = c + 8 * ix[0], *p1 = c + 8 * ix[1],
                 *p2 = c + 8 * ix[2], *p3 = c + 8 * ix[3];
    // 4x4 transposes of the energy halves and the virial halves.
    Vec cf[8];
    for (int h = 0; h < 2; ++h) {
      const Vec a0 = _mm256_loadu_pd(p0 + 4 * h);
      const Vec a1 = _mm256_loadu_pd(p1 + 4 * h);
      const Vec a2 = _mm256_loadu_pd(p2 + 4 * h);
      const Vec a3 = _mm256_loadu_pd(p3 + 4 * h);
      const Vec t0 = _mm256_unpacklo_pd(a0, a1);
      const Vec t1 = _mm256_unpackhi_pd(a0, a1);
      const Vec t2 = _mm256_unpacklo_pd(a2, a3);
      const Vec t3 = _mm256_unpackhi_pd(a2, a3);
      cf[4 * h + 0] = _mm256_permute2f128_pd(t0, t2, 0x20);
      cf[4 * h + 1] = _mm256_permute2f128_pd(t1, t3, 0x20);
      cf[4 * h + 2] = _mm256_permute2f128_pd(t0, t2, 0x31);
      cf[4 * h + 3] = _mm256_permute2f128_pd(t1, t3, 0x31);
    }
    en = Fma(f, Fma(f, Fma(f, cf[3], cf[2]), cf[1]), cf[0]);
    vir = Fma(f, Fma(f, Fma(f, cf[7], cf[6]), cf[5]), cf[4]);
  }
};
#endif

// ------------------------------------------------------------- AVX-512 ----

#if !defined(GOMC_SIMD_GENERIC) && defined(__AVX512F__)
struct Avx512 {
  static constexpr int W = 8;
  static const char *Name() { return "AVX-512"; }
  typedef __m512d Vec;
  typedef __mmask8 Mask;

  static Vec Set1(const double a) { return _mm512_set1_pd(a); }
  static Vec Load(const double *p) { return _mm512_loadu_pd(p); }
  static Vec LoadJ4(const double *p) {
    return _mm512_broadcast_f64x4(_mm256_loadu_pd(p));
  }
  static Vec LoadRows(const double *const *row) {
    return _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_loadu_pd(row[0])),
                              _mm256_loadu_pd(row[1]), 1);
  }
  static void Store(double *p, const Vec a) { _mm512_storeu_pd(p, a); }
  static Vec Add(const Vec a, const Vec b) { return _mm512_add_pd(a, b); }
  static Vec Sub(const Vec a, const Vec b) { return _mm512_sub_pd(a, b); }
  static Vec Mul(const Vec a, const Vec b) { return _mm512_mul_pd(a, b); }
  static Vec Div(const Vec a, const Vec b) { return _mm512_div_pd(a, b); }
  static Vec Fma(const Vec a, const Vec b, const Vec c) {
    return _mm512_fmadd_pd(a, b, c);
  }
  static Vec Rint(const Vec a) {
    return _mm512_roundscale_pd(a,
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  }
  static Mask Lt(const Vec a, const Vec b) {
    return _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ);
  }
  static Mask Le(const Vec a, const Vec b) {
    return _mm512_cmp_pd_mask(a, b, _CMP_LE_OQ);
  }
  static Mask Ge(const Vec a, const Vec b) {
    return _mm512_cmp_pd_mask(a, b, _CMP_GE_OQ);
  }
  static Mask Ne(const Vec a, const Vec b) {
    return _mm512_cmp_pd_mask(a, b, _CMP_NEQ_OQ);
  }
  static Mask And(const Mask a, const Mask b) { return a & b; }
  static Mask AndNot(const Mask a, const Mask b) { return a & (Mask)~b; }
  static Mask Or(const Mask a, const Mask b) { return a | b; }
  static Mask None() { return 0; }
  static bool Any(const Mask a) { return a != 0; }
  static Vec Select(const Mask m, const Vec a, const Vec b) {
    return _mm512_mask_blend_pd(m, b, a);
  }
  static double HSum(const Vec a) { return _mm512_reduce_add_pd(a); }
  //! Lanes k and k + 4 hold the same j-atom.
  static void FoldJ4Add(double *out, const Vec a) {
    const __m256d s = _mm256_add_pd(_mm512_castpd512_pd256(a),
                                    _mm512_extractf64x4_pd(a, 1));
    _mm256_storeu_pd(out, _mm256_add_pd(_mm256_loadu_pd(out), s));
  }
  static Vec Gather(const double *base, const int *idx) {
    return _mm512_i32gather_pd(_mm256_loadu_si256((const __m256i *)idx), base,
                               8);
  }
  static void TableEnVir(const double *c, const Vec t, Vec &en, Vec &vir) {
    const __m256i i8 = _mm512_cvttpd_epi32(t);
    const Vec f = _mm512_sub_pd(t, _mm512_cvtepi32_pd(i8));
    alignas(32) int ix[8];
    _mm256_store_si256((__m256i *)ix, i8);
    // Row k is lane k's whole line: coefficients 0-3 energy, 4-7 virial.
    Vec r[8];
    for (int k = 0; k < 8; ++k)
      r[k] = _mm512_loadu_pd(c + 8 * ix[k]);
    // 8x8 transpose: unpack pairs of rows, then two rounds of 128-bit lane
    // shuffles. 0x88 takes 128-bit lanes (0, 2) of each source, 0xDD (1, 3).
    const Vec t0 = _mm512_unpacklo_pd(r[0], r[1]);
    const Vec t1 = _mm512_unpackhi_pd(r[0], r[1]);
    const Vec t2 = _mm512_unpacklo_pd(r[2], r[3]);
    const Vec t3 = _mm512_unpackhi_pd(r[2], r[3]);
    const Vec t4 = _mm512_unpacklo_pd(r[4], r[5]);
    const Vec t5 = _mm512_unpackhi_pd(r[4], r[5]);
    const Vec t6 = _mm512_unpacklo_pd(r[6], r[7]);
    const Vec t7 = _mm512_unpackhi_pd(r[6], r[7]);
    const Vec u0 = _mm512_shuffle_f64x2(t0, t2, 0x88);
    const Vec u1 = _mm512_shuffle_f64x2(t0, t2, 0xDD);
    const Vec u2 = _mm512_shuffle_f64x2(t4, t6, 0x88);
    const Vec u3 = _mm512_shuffle_f64x2(t4, t6, 0xDD);
    const Vec v0 = _mm512_shuffle_f64x2(t1, t3, 0x88);
    const Vec v1 = _mm512_shuffle_f64x2(t1, t3, 0xDD);
    const Vec v2 = _mm512_shuffle_f64x2(t5, t7, 0x88);
    const Vec v3 = _mm512_shuffle_f64x2(t5, t7, 0xDD);
    const Vec c0 = _mm512_shuffle_f64x2(u0, u2, 0x88);
    const Vec c4 = _mm512_shuffle_f64x2(u0, u2, 0xDD);
    const Vec c2 = _mm512_shuffle_f64x2(u1, u3, 0x88);
    const Vec c6 = _mm512_shuffle_f64x2(u1, u3, 0xDD);
    const Vec c1 = _mm512_shuffle_f64x2(v0, v2, 0x88);
    const Vec c5 = _mm512_shuffle_f64x2(v0, v2, 0xDD);
    const Vec c3 = _mm512_shuffle_f64x2(v1, v3, 0x88);
    const Vec c7 = _mm512_shuffle_f64x2(v1, v3, 0xDD);
    en = Fma(f, Fma(f, Fma(f, c3, c2), c1), c0);
    vir = Fma(f, Fma(f, Fma(f, c7, c6), c5), c4);
  }
};
typedef Avx512 Native;
#elif !defined(GOMC_SIMD_GENERIC) && defined(__AVX2__) && defined(__FMA__)
typedef Avx2 Native;
#else
typedef Generic Native;
#endif

} // namespace simd

#endif /*SIMD_LAYER_H*/
