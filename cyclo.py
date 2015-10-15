# ffo.py
# Public Domain
# Provided by Leo Ducas and Thomas Prest
# Reference implementation of algorithms proposed in
# [Fast Fourier Orthogonalization, Ducas and Prest]
# To be used for testing purposes only

from numpy.fft import fft,ifft
from numpy import zeros

### Embed an element of Z[X]/(X^n+1) into Z[X]/(X^2n - 1)
### Using multiplication by the idempotent element e

def embedding(f):
	d = 2*len(f)
	r = zeros(d)
	r[0:d/2] = f
	e = zeros(d)
	e[0] = .5
	e[d/2] = -.5
	return ifft(fft(r)*fft(e)).real