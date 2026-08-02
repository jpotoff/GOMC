#include <cmath>
#include <vector>

#pragma omp declare simd
inline double erfc_cody(double x) {
    return std::exp(-x);
}

void calc(const std::vector<double>& in, std::vector<double>& out) {
    #pragma omp simd
    for(int i=0; i<in.size(); ++i) {
        out[i] = erfc_cody(in[i]);
    }
}
