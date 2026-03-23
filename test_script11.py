import numpy as np
# Let's verify FFTW c2r with EXACT python equivalent of what C++ code does.
Kx, Ky, Kz = 4, 4, 4
q = np.random.rand()
w = np.random.rand(Kx, Ky, Kz)

r2c_out = np.fft.rfftn(w)
G = np.random.rand(*r2c_out.shape)

S_trial = G * r2c_out

# c2r is inverse transform (UNNORMALIZED)
c2r_out = np.fft.irfftn(S_trial, s=(Kx, Ky, Kz)) * (Kx * Ky * Kz)

# Interpolate:
interp = np.sum(w * c2r_out)

print("true dot product:", interp)

# Compute true cross term 
# sum_{Full} G S \Delta S^*
w_full = np.fft.fftn(w)
G_full = np.zeros((Kx, Ky, Kz))
# Just map half space cleanly
G_full = np.fft.fftn(np.fft.irfftn(G, s=(Kx,Ky,Kz))*Kx*Ky*Kz) / (Kx*Ky*Kz)
# This isn't symmetric. Let's build explicit symmetric G.

