import numpy as np

N = 8
Q = np.random.rand(N)
S = np.fft.rfft(Q)

# G is physically derived from energy = 0.5 * sum G |S|^2.
# So G |S|^2 has physical units of Energy.
E_init = 0.5 * np.sum(np.random.rand(len(S)) * np.abs(S)**2)

# If we compute V = irfft(G * S) * N, 
# then \sum Q * V = \sum_m G |S|^2 = 2 * E_init.
# Is this TRUE?

G = np.random.rand(len(S))
E_init = 0.5 * np.sum(G * np.abs(S)**2) * 2.0 # 2.0 for approx weight
V_unnorm = np.fft.irfft(G * S) * N
cross = np.sum(Q * V_unnorm)

print("2 * E_init =", 2 * E_init)
print("cross      =", cross)

# Wait. What if GOMC's green function factor "fac" already incorporates 1/V = 1 / (N * V_voxel) ??
# fac = 4.0 * M_PI * num::qqFact / volume;
# In continuous PME, E = 1/(2V) \sum 4\pi/k^2 |S_cont(k)|^2.
# In discrete PME, S_discrete(m) = \sum Q_grid(r) e^{-ikr}.
# Does S_discrete MATCH S_cont?
# S_cont(k) = \sum q_j e^{-ik r_j}.
# S_discrete(m) = \sum Q_grid(r) e^{-ikr} 
#               = \sum_r e^{-ikr} \sum_j q_j W(r - r_j)
#               = \sum_j q_j \sum_r W(r - r_j) e^{-ikr}
#               = \sum_j q_j e^{-ikr_j} \sum_r W(r) e^{-ikr}
#               = S_cont(k) * B(m).
# YES! S_discrete EXACTLY equals S_cont * B(m).
# So S_discrete is INDEPENDENT of N in magnitude!
# If so, E = 1/(2V) \sum 4\pi/k^2 |S_discrete(k)|^2 / B(m)^2.
# In GOMC, Gptr[i] = BFunc * expTerm * fac / kSq.
# So Gptr = 4\pi / (V k^2) * B * e^{-k^2/4\alpha}.
# So \sum G_m |S_m|^2 evaluates EXACTLY to 2 * E_init !
# And it evaluates this WITHOUT ANY DEPENDENCE ON N!

print("S_discrete is independent of N. G_m is independent of N.")
print("So V_m = G_m * S_m is independent of N.")
print("But V_unnorm = irfft(V_m) * N is PROPORTIONAL TO N !!!!!!!")
