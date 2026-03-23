import numpy as np

N = 8
Q = np.random.rand(N, N, N)
dQ = np.random.rand(N, N, N) - 0.5

S_ref_full = np.fft.fftn(Q)
dS_full = np.fft.fftn(dQ)

G_full = np.zeros((N, N, N))
for i in range(N):
    for j in range(N):
        for k in range(N):
            ii = i if i <= N//2 else i - N
            jj = j if j <= N//2 else j - N
            kk = k if k <= N//2 else k - N
            G_full[i,j,k] = np.exp(-(ii**2 + jj**2 + kk**2)/10.0)

E_init = 0.5 * np.sum(G_full * np.abs(S_ref_full)**2)
E_new = 0.5 * np.sum(G_full * np.abs(S_ref_full + dS_full)**2)
true_dE = E_new - E_init

S_ref = np.fft.rfftn(Q)
dS = np.fft.rfftn(dQ)

weight = np.ones_like(S_ref) * 2.0
weight[..., 0] = 1.0 # mz = 0 plane
if N % 2 == 0:
    weight[..., N//2] = 1.0
    
G_half = G_full[..., :N//2+1]
dE_self = 0.5 * np.sum(weight * G_half * np.abs(dS)**2)

V_m = G_half * S_ref
V_r = np.fft.irfftn(V_m, s=(N, N, N)) * (N**3)
dE_back = np.sum(dQ * V_r)

calc_dE = dE_back + dE_self

print(f"true_dE: {true_dE}")
print(f"calc_dE: {calc_dE}")

