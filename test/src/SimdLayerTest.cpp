/*******************************************************************************
Tests for the SIMD layer the cluster-pair kernel is written against
(SimdLayer.h).

Each primitive of simd::Native -- whichever backend this build selected:
AVX-512, AVX2 or the generic one -- is compared lane by lane with the plain
scalar expression it stands for. On a machine without AVX the generic backend
is what runs; the vector backends are covered wherever the build targets
them, which is where their results matter.

TableEnVir gets the most attention: it loads each lane's 8-coefficient line
and transposes the lines in registers, so a wrong shuffle constant would
silently pair one lane's distance with another lane's coefficients.
*******************************************************************************/

#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "SimdLayer.h"

namespace {

typedef simd::Native B;
constexpr int W = B::W;

std::vector<double> Random(std::mt19937 &g, const int n, const double lo,
                           const double hi) {
  std::uniform_real_distribution<double> u(lo, hi);
  std::vector<double> v(n);
  for (double &x : v)
    x = u(g);
  return v;
}

std::vector<double> Lanes(const B::Vec &a) {
  std::vector<double> v(W);
  B::Store(v.data(), a);
  return v;
}

//! Lane k of a mask, read through Select.
std::vector<bool> Lanes(const B::Mask &m) {
  const std::vector<double> v =
      Lanes(B::Select(m, B::Set1(1.0), B::Set1(0.0)));
  std::vector<bool> r(W);
  for (int k = 0; k < W; ++k)
    r[k] = v[k] == 1.0;
  return r;
}

} // namespace

TEST(SimdLayerTest, Arithmetic) {
  std::mt19937 g(11);
  for (int rep = 0; rep < 100; ++rep) {
    const std::vector<double> a = Random(g, W, -10.0, 10.0);
    const std::vector<double> b = Random(g, W, 0.5, 10.0);
    const std::vector<double> c = Random(g, W, -10.0, 10.0);
    const B::Vec va = B::Load(a.data()), vb = B::Load(b.data()),
                 vc = B::Load(c.data());
    const std::vector<double> add = Lanes(B::Add(va, vb)),
                              sub = Lanes(B::Sub(va, vb)),
                              mul = Lanes(B::Mul(va, vb)),
                              div = Lanes(B::Div(va, vb)),
                              fma = Lanes(B::Fma(va, vb, vc));
    for (int k = 0; k < W; ++k) {
      EXPECT_EQ(add[k], a[k] + b[k]);
      EXPECT_EQ(sub[k], a[k] - b[k]);
      EXPECT_EQ(mul[k], a[k] * b[k]);
      EXPECT_EQ(div[k], a[k] / b[k]);
      // Fused or not, depending on the backend.
      EXPECT_NEAR(fma[k], a[k] * b[k] + c[k], 1e-13);
    }
    EXPECT_NEAR(B::HSum(va), [&] {
      double s = 0.0;
      for (int k = 0; k < W; ++k)
        s += a[k];
      return s;
    }(), 1e-13);
  }
}

TEST(SimdLayerTest, RintRoundsHalfToEven) {
  // The minimum image relies on rint's round-half-to-even, as
  // MinImageSigned does.
  const double in[8] = {2.5, -2.5, 3.5, -0.5, 0.49999999999, 1.5, -1.5, 7.0};
  const double out[8] = {2.0, -2.0, 4.0, -0.0, 0.0, 2.0, -2.0, 7.0};
  for (int base = 0; base < 8; base += W) {
    const std::vector<double> r = Lanes(B::Rint(B::Load(in + base)));
    for (int k = 0; k < W; ++k)
      EXPECT_EQ(r[k], out[base + k]) << "input " << in[base + k];
  }
}

TEST(SimdLayerTest, MasksAndSelect) {
  std::mt19937 g(12);
  for (int rep = 0; rep < 100; ++rep) {
    std::vector<double> a = Random(g, W, -1.0, 1.0);
    std::vector<double> b = Random(g, W, -1.0, 1.0);
    b[0] = a[0]; // exercise equality
    const B::Vec va = B::Load(a.data()), vb = B::Load(b.data());
    const std::vector<bool> lt = Lanes(B::Lt(va, vb)), le = Lanes(B::Le(va, vb)),
                            ge = Lanes(B::Ge(va, vb)), ne = Lanes(B::Ne(va, vb));
    const std::vector<bool> andm = Lanes(B::And(B::Lt(va, vb), B::Ne(va, vb)));
    const std::vector<bool> orm = Lanes(B::Or(B::Lt(va, vb), B::Ge(va, vb)));
    const std::vector<bool> andnot =
        Lanes(B::AndNot(B::Le(va, vb), B::Lt(va, vb)));
    const std::vector<double> sel = Lanes(B::Select(B::Lt(va, vb), va, vb));
    bool anyLt = false;
    for (int k = 0; k < W; ++k) {
      EXPECT_EQ(lt[k], a[k] < b[k]);
      EXPECT_EQ(le[k], a[k] <= b[k]);
      EXPECT_EQ(ge[k], a[k] >= b[k]);
      EXPECT_EQ(ne[k], a[k] != b[k]);
      EXPECT_EQ(andm[k], a[k] < b[k] && a[k] != b[k]);
      EXPECT_TRUE(orm[k]);
      EXPECT_EQ(andnot[k], a[k] == b[k]);
      EXPECT_EQ(sel[k], a[k] < b[k] ? a[k] : b[k]);
      anyLt = anyLt || a[k] < b[k];
    }
    EXPECT_EQ(B::Any(B::Lt(va, vb)), anyLt);
    unsigned bits = 0;
    for (int k = 0; k < W; ++k)
      bits |= (unsigned)(a[k] < b[k]) << k;
    EXPECT_EQ(B::Bits(B::Lt(va, vb)), bits);
  }
  EXPECT_FALSE(B::Any(B::None()));
}

TEST(SimdLayerTest, TileLayout) {
  // LoadJ4 repeats 4 values across the register; FoldJ4Add sums the lanes
  // holding the same j, lane l being j = l % 4.
  const double p[4] = {1.0, 2.0, 3.0, 4.0};
  const std::vector<double> j4 = Lanes(B::LoadJ4(p));
  for (int k = 0; k < W; ++k)
    EXPECT_EQ(j4[k], p[k % 4]);

  std::vector<double> v(W);
  for (int k = 0; k < W; ++k)
    v[k] = 1.0 + k * 10.0;
  double out[4] = {0.5, 0.5, 0.5, 0.5};
  B::FoldJ4Add(out, B::Load(v.data()));
  for (int j = 0; j < 4; ++j) {
    double expect = 0.5;
    for (int k = j; k < W; k += 4)
      expect += v[k];
    EXPECT_EQ(out[j], expect);
  }
}

TEST(SimdLayerTest, LoadRows) {
  // Each group of 4 lanes (one i of the tile) comes from its own row, as
  // the kernel's per-kind LJ rows are read.
  const double a[4] = {1.0, 2.0, 3.0, 4.0};
  const double b[4] = {5.0, 6.0, 7.0, 8.0};
  const double *const rows[2] = {a, b};
  const std::vector<double> r = Lanes(B::LoadRows(rows));
  for (int k = 0; k < W; ++k)
    EXPECT_EQ(r[k], rows[k / 4][k % 4]);
}

TEST(SimdLayerTest, Gather) {
  std::mt19937 g(13);
  const std::vector<double> base = Random(g, 64, -1.0, 1.0);
  std::uniform_int_distribution<int> u(0, 63);
  for (int rep = 0; rep < 50; ++rep) {
    int idx[8];
    for (int k = 0; k < 8; ++k)
      idx[k] = u(g);
    const std::vector<double> r = Lanes(B::Gather(base.data(), idx));
    for (int k = 0; k < W; ++k)
      EXPECT_EQ(r[k], base[idx[k]]);
  }
}

TEST(SimdLayerTest, TableEnVirMatchesScalarCubics) {
  // A table of random coefficients, 8 per interval as in EwaldRealTable:
  // [0..3] the energy cubic, [4..7] the virial cubic.
  std::mt19937 g(14);
  const int n = 200;
  const std::vector<double> c = Random(g, 8 * n, -2.0, 2.0);
  for (int rep = 0; rep < 500; ++rep) {
    const std::vector<double> t = Random(g, W, 0.0, n - 1.0);
    B::Vec en, vir;
    B::TableEnVir(c.data(), B::Load(t.data()), en, vir);
    const std::vector<double> e = Lanes(en), v = Lanes(vir);
    for (int k = 0; k < W; ++k) {
      const int i = (int)t[k];
      const double f = t[k] - i;
      const double *p = c.data() + 8 * i;
      EXPECT_NEAR(e[k], p[0] + f * (p[1] + f * (p[2] + f * p[3])), 1e-13)
          << "lane " << k;
      EXPECT_NEAR(v[k], p[4] + f * (p[5] + f * (p[6] + f * p[7])), 1e-13)
          << "lane " << k;
    }
  }
}

TEST(SimdLayerTest, ReportsBackend) {
  std::cout << "SIMD backend: " << B::Name() << ", " << W << " lanes"
            << std::endl;
  SUCCEED();
}
