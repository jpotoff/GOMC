import numpy as np

# Let's test if FFTW's definition of C2R and R2C introduces an unexpected factor of 2 or N
# In particular, numpy RFFT computes standard DFT. Let's assume C2R computes unnormalized IFFT.
print("Checking normalization...")
