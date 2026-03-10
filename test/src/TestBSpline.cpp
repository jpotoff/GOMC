#include "BSpline.h"
#include "gtest/gtest.h"
#include <cmath>
#include <vector>

using namespace bspline;

TEST(BSpline, PartitionOfUnity) {
  for (int order : {4, 6, 8}) {
    for (double u = 0.0; u <= 1.0; u += 0.1) {
      std::vector<double> vals(order);
      EvalAll(order, u, vals.data());
      double sum = 0.0;
      for (int i = 0; i < order; ++i) {
        sum += vals[i];
      }
      EXPECT_NEAR(sum, 1.0, 1e-10);
    }
  }
}

TEST(BSpline, Positivity) {
  for (int order : {4, 6, 8}) {
    for (double u = 0.0; u <= 1.0; u += 0.1) {
      std::vector<double> vals(order);
      EvalAll(order, u, vals.data());
      for (int i = 0; i < order; ++i) {
        EXPECT_GE(vals[i], 0.0);
      }
    }
  }
}

TEST(BSpline, Support) {
  for (int order : {4, 6, 8}) {
    EXPECT_EQ(Eval(order, -0.1), 0.0);
    EXPECT_EQ(Eval(order, order + 0.1), 0.0);
  }
}

TEST(BSpline, DerivativeFiniteDiff) {
  double h = 1e-6;
  for (int order : {4, 6, 8}) {
    for (double u = 0.5; u < order - 0.5; u += 0.5) {
      double deriv = EvalDeriv(order, u);
      double fd = (Eval(order, u + h) - Eval(order, u - h)) / (2.0 * h);
      EXPECT_NEAR(deriv, fd, 1e-5);
    }
  }
}

TEST(BSpline, EvalAllDerivMatchesDeriv) {
  for (int order : {4, 6, 8}) {
    for (double f = 0.1; f < 1.0; f += 0.2) {
      std::vector<double> derivs(order);
      EvalAllDeriv(order, f, derivs.data());
      for (int i = 0; i < order; ++i) {
        EXPECT_NEAR(derivs[i], EvalDeriv(order, f + i), 1e-10);
      }
    }
  }
}

TEST(BSpline, ModuliNonZero) {
  for (int order : {4, 6, 8}) {
    auto mod = BSplineModuli(64, order);
    for (double m : mod) {
      EXPECT_GT(m, 0.0);
    }
  }
}
