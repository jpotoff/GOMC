import numpy as np

N = 8
Q = np.random.rand(N, N, N)
dQ = np.random.rand(N, N, N) - 0.5

# Forward RFFT: real to half-complex
S_ref = np.fft.rfftn(Q)
dS = np.fft.rfftn(dQ)

# G is real, symmetric
G = np.random.rand(*S_ref.shape)

# Create full 3D FFT for true E
S_ref_full = np.fft.fftn(Q)
dS_full = np.fft.fftn(dQ)
G_full = np.random.rand(N, N, N)

# To make G Hermitian-symmetric (G(-m) = G(m))
G_full = (G_full + np.conj(np.flip(np.flip(np.flip(G_full, 0), 1), 2))) / 2.0
G_full = np.real(G_full)

E_init = 0.5 * np.sum(G_full * np.abs(S_ref_full)**2)
E_new = 0.5 * np.sum(G_full * np.abs(S_ref_full + dS_full)**2)
true_dE = E_new - E_init

# In half-space, we evaluate self-energy sum exactly like GOMC's SumMeshEnergy:
weight = np.ones_like(S_ref) * 2.0
weight[..., 0] = 1.0 # mz = 0 plane
if N % 2 == 0:
    weight[..., N//2] = 1.0
    
# Extract corresponding half-space G:
G_half = G_full[..., :N//2+1]
dE_self = 0.5 * np.sum(weight * G_half * np.abs(dS)**2)

# Unnormalized C2R (like FFTW) -> equivalent to N^3 * irfftn
V_m = G_half * S_ref
V_r = np.fft.irfftn(V_m, s=(N, N, N)) * (N**3)
dE_back = np.sum(dQ * V_r)

calc_dE = dE_back + dE_self

print(f"true_dE: {true_dE}")
print(f"calc_dE: {calc_dE}")
