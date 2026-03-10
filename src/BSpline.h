#pragma once
#include <vector>

namespace bspline {
double Eval(int order, double u);
double EvalDeriv(int order, double u);
void EvalAll(int order, double u, double *vals);
void EvalAllDeriv(int order, double u, double *d);
std::vector<double> BSplineModuli(int K, int order);
} // namespace bspline
