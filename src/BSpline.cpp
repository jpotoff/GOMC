#include "BSpline.h"
#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace bspline {

double Eval(int order, double u) {
  if (u < 0.0 || u > order)
    return 0.0;
  if (order == 2) {
    if (u < 1.0)
      return u;
    if (u < 2.0)
      return 2.0 - u;
    return 0.0;
  }
  double val = 0.0;
  val += (u / (order - 1.0)) * Eval(order - 1, u);
  val += ((order - u) / (order - 1.0)) * Eval(order - 1, u - 1.0);
  return val;
}

double EvalDeriv(int order, double u) {
  if (u < 0.0 || u > order)
    return 0.0;
  return Eval(order - 1, u) - Eval(order - 1, u - 1.0);
}

void EvalAll(int order, double f, double *vals) {
  vals[0] = 1.0;
  for (int n = 2; n <= order; ++n) {
    double saved = 0.0;
    double inv_n_1 = 1.0 / (n - 1.0);
    for (int i = 0; i < n - 1; ++i) {
      double u = f + i;
      double term1 = u * inv_n_1 * vals[i];
      double term2 = (n - u - 1.0) * inv_n_1 * vals[i];
      vals[i] = saved + term1;
      saved = term2;
    }
    vals[n - 1] = saved;
  }
}

void EvalAllDeriv(int order, double f, double *d) {
  std::vector<double> vals(order, 0.0);
  vals[0] = 1.0;
  for (int n = 2; n <= order - 1; ++n) {
    double saved = 0.0;
    double inv_n_1 = 1.0 / (n - 1.0);
    for (int i = 0; i < n - 1; ++i) {
      double u = f + i;
      double term1 = u * inv_n_1 * vals[i];
      double term2 = (n - u - 1.0) * inv_n_1 * vals[i];
      vals[i] = saved + term1;
      saved = term2;
    }
    vals[n - 1] = saved;
  }

  d[0] = vals[0] - 0.0;
  for (int i = 1; i < order; ++i) {
    d[i] = vals[i] - vals[i - 1];
  }
}

std::vector<double> BSplineModuli(int K, int order) {
  std::vector<double> moduli(K, 0.0);
  for (int m = 0; m < K; ++m) {
    double sum_cos = 0.0;
    double sum_sin = 0.0;
    double arg = 2.0 * M_PI * m / K;
    for (int k = 0; k < order - 1; ++k) {
      double mk = Eval(order, k + 1.0);
      sum_cos += mk * std::cos(arg * k);
      sum_sin += mk * std::sin(arg * k);
    }
    double b2 = sum_cos * sum_cos + sum_sin * sum_sin;
    if (b2 == 0.0) {
      moduli[m] = 0.0;
    } else {
      moduli[m] = 1.0 / b2;
    }
  }
  return moduli;
}

} // namespace bspline
