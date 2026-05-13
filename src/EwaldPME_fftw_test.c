#include <stdio.h>
#include <fftw3.h>

int main() {
    // Some sizes, like 45, are known to force out-of-place destruction
    // in FFTW3 for multi-dimensional r2c depending on the system.
    int N = 45; 
    double *in = fftw_malloc(sizeof(double) * N * N * N);
    fftw_complex *out = fftw_malloc(sizeof(fftw_complex) * N * N * (N / 2 + 1));
    for (int i = 0; i < N*N*N; i++) in[i] = i;
    
    // Test without PRESERVE_INPUT
    fftw_plan p = fftw_plan_dft_r2c_3d(N, N, N, in, out, FFTW_ESTIMATE);
    
    // reset array to ensure it's clean before execute
    for (int i = 0; i < N*N*N; i++) in[i] = i;
    
    fftw_execute(p);
    
    int errors = 0;
    for (int i = 0; i < N*N*N; i++) {
        if (in[i] != i) errors++;
    }
    printf("Without PRESERVE_INPUT: %d elements were modified!\n", errors);
    
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
    return 0;
}
