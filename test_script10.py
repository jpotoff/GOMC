import numpy as np

# Let's verify parseval's theorem again VERY carefully, including the weight factors.
N = 6
Kz = N
Ky = N
Kx = N

Q = np.random.rand(Kx, Ky, Kz)
dQ = np.random.rand(Kx, Ky, Kz) - 0.5

# Forward
S = np.fft.rfftn(Q)
dS = np.fft.rfftn(dQ)

G = np.random.rand(*S.shape)

weight = np.ones_like(S) * 2.0
weight[..., 0] = 1.0
if Kz % 2 == 0:
    weight[..., Kz//2] = 1.0

E_init = 0.5 * np.sum(weight * G * np.abs(S)**2)
E_new = 0.5 * np.sum(weight * G * np.abs(S + dS)**2)
true_dE = E_new - E_init

# Self
dE_self = 0.5 * np.sum(weight * G * np.abs(dS)**2)

# Cross
H = G * S
V = np.fft.irfftn(H, s=(Kx, Ky, Kz)) * (Kx * Ky * Kz)
dE_back = np.sum(dQ * V)

print("true dE:", true_dE)
print("dE_back + dE_self:", dE_back + dE_self)
print("Ratio:", true_dE / (dE_back + dE_self))

