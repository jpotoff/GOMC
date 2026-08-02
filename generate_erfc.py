import math
import numpy as np
from numpy.polynomial import Chebyshev, Polynomial

def fit_single_interval(degree):
    x = np.linspace(0, 4.0, 100000)
    y = np.array([math.erfc(xt) * math.exp(xt * xt) for xt in x])
    
    # Fit Chebyshev polynomial
    c = Chebyshev.fit(x, y, degree)
    p = c.convert(kind=Polynomial)
    
    y_fit = p(x) * np.exp(-x**2)
    y_target = np.array([math.erfc(xt) for xt in x])
    err = np.max(np.abs(y_fit - y_target))
    
    print(f"// Degree {degree}, Max error: {err}")
    print("double c[] = {")
    for coef in p.coef:
        print(f"    {coef:.17g},")
    print("};")

fit_single_interval(25)

