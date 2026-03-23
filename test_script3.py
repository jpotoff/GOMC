import numpy as np

N = 8
Q_r = np.random.rand(N)
dQ_r = np.random.rand(N) - 0.5

S_ref = np.fft.rfft(Q_r)
dS = np.fft.rfft(dQ_r)

G = np.random.rand(len(S_ref))

# Full cross term via full FFT
S_ref_full = np.fft.fft(Q_r)
dS_full = np.fft.fft(dQ_r)
G_full = np.zeros(N)
for i in range(N):
    if i <= N//2:
        G_full[i] = G[i]
    else:
        G_full[i] = G[N-i]

true_cross = np.sum(G_full * np.real(S_ref_full * np.conj(dS_full)))

# Calculate via irfft
V_m = G * S_ref
V_r = np.fft.irfft(V_m, n=N) * N  # FFTW unnormalized irfft
calc_cross = np.sum(dQ_r * V_r)

print(f"true_cross: {true_cross}")
print(f"calc_cross: {calc_cross}")

