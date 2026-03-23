import numpy as np

N = 8
x = np.random.rand(N)
y = np.random.rand(N)

X = np.fft.fft(x)
Y = np.fft.fft(y)

# We want sum_m X(m) Y*(m)
target = np.sum(X * np.conj(Y))

# Math Parseval: sum_x x * y = 1/N * sum_m X * Y*
print("Math Parseval target:", N * np.sum(x * y))
print("Actual target:", target)

# FFTW iFFT:
X_fftw_ifft = np.fft.ifft(X) * N
cross_fftw = np.sum(X_fftw_ifft * y)

print("Target via FFTW iFFT:", cross_fftw)
