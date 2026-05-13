#include <stdio.h>
#include <fftw3.h>

int main() {
    int N = 4;
    double *in = fftw_malloc(sizeof(double) * N * N * N);
    fftw_complex *out = fftw_malloc(sizeof(fftw_complex) * N * N * (N / 2 + 1));
    for (int i = 0; i < N*N*N; i++) in[i] = i;
    fftw_plan p = fftw_plan_dft_r2c_3d(N, N, N, in, out, FFTW_ESTIMATE);
    
    // Check if plan creation modified it
    printf("Before plan: %.1f\n", in[1]);
    for (int i = 0; i < N*N*N; i++) in[i] = i;
    
    fftw_execute(p);
    printf("After execute: %.1f\n", in[1]);
    
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
    return 0;
}
