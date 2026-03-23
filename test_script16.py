import numpy as np

def V_center(N):
    # A charge at origin
    Q = np.zeros(N)
    Q[0] = 1.0
    S = np.fft.fft(Q)
    
    # physical G
    G = np.zeros(N)
    for m in range(N):
        k = m if m <= N/2 else m - N
        if k == 0: continue
        ksq = k**2
        G[m] = np.exp(-ksq / 10.0) / ksq
        
    V_m = G * S
    V_r = np.fft.ifft(V_m).real * N
    # The value of potential at r = 1
    return V_r[1]

print("N = 16:", V_center(16))
print("N = 32:", V_center(32))
print("N = 64:", V_center(64))
print("N = 128:", V_center(128))
