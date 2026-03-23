import numpy as np
# Let's create an exact replica of EwaldPME algorithm on a 64^3 grid
N = 32
vol = 1000.0 # random Box volume
coords = np.random.rand(10, 3) * (N * 0.9)
charges = np.ones(10)
charges[-5:] = -1.0 # neutral

mesh = np.zeros((N, N, N))
for i in range(10):
    ix, iy, iz = int(coords[i,0]), int(coords[i,1]), int(coords[i,2])
    mesh[ix, iy, iz] += charges[i]

S = np.fft.rfftn(mesh)

G = np.zeros(S.shape)
for i in range(N):
    for j in range(N):
        for k in range(N//2+1):
            kx = i if i <= N//2 else i - N
            ky = j if j <= N//2 else j - N
            kz = k
            ksq = kx**2 + ky**2 + kz**2
            if ksq == 0: continue
            # very crude G
            G[i,j,k] = np.exp(-ksq / 10.0) / ksq

weight = np.ones(S.shape) * 2.0
weight[..., 0] = 1.0
if N % 2 == 0: weight[..., N//2] = 1.0
E_init = 0.5 * np.sum(weight * G * np.abs(S)**2)

# Now standard potential 
V_m = G * S
V_r = np.fft.irfftn(V_m, s=(N,N,N)) * (N**3)

E_real = 0.5 * np.sum(mesh * V_r)
print("E_init:", E_init)
print("E_real:", E_real)
