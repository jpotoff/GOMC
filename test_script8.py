import numpy as np

# Is there a factor of N in FFTW c2r vs r2c?
x = np.random.rand(8)
X = np.fft.rfft(x)
y = np.fft.irfft(X) * 8
print("Max diff:", np.max(np.abs(x*8 - y)))

