import numpy as np

N = 8
x = np.random.rand(N)
S = np.fft.fft(x)
G = np.random.rand(N)

# energy in recip space = sum_m G(m) * |S(m)|^2
recip_energy = np.sum(G * np.abs(S)**2)

# V(m) = G(m) * S(m)
V_m = G * S

# ifft in numpy divides by N: ifft(X) = 1/N * sum X exp(+)
# FFTW c2r computes unnormalized inverse FFT: x_out = sum X exp(+)
# So FFTW_c2r(V_m) = N * ifft(V_m)
V_r_fftw = np.fft.ifft(V_m) * N

real_energy = np.sum(x * V_r_fftw)

print(f"recip_energy: {recip_energy}")
print(f"real_energy (FFTW unnormalized): {real_energy}")
print(f"real_energy (Normalized by N): {real_energy / N}")

