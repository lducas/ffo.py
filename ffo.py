# ffo.py
# Public Domain
# Provided by Leo Ducas and Thomas Prest
# Reference implementation of algorithms proposed in
# [Fast Fourier Orthogonalization, Ducas and Prest]
# To be used for testing purposes only

from math import sqrt
from math import isnan as isNaN

import random
import numpy
import sys

from numpy.fft import rfft,irfft,fft,ifft
from numpy import inner, zeros,sign, conjugate,exp,pi
from numpy import roll,transpose,diag
from numpy.linalg import qr, cholesky,inv
from numpy import matrix, vectorize
from numpy import array as vector

from verif import create_verifier, verif_babai

# Inverse vectorize operation V^-1, i/o in fft format
def ffmerge((F1,F2)):
	d = 2*len(F1)
	F = 0.j*zeros(d)
	w = exp(-2.j*pi / d)
	W = vector([w**i for i in range(d/2)])
	F[:d/2] = F1 + W* F2
	F[d/2:] = F1 - W* F2
	return F

# Vectorize operation V, i/o in fft format
def ffsplit(F):
	d = len(F)
	winv = exp(2.j*pi / d)
	Winv = vector([winv**i for i in range(d/2)])
	F1 = .5* (F[:d/2] + F[d/2:])
	F2 = .5* (F[:d/2] - F[d/2:]) * Winv
	return (F1,F2)

# ffLDL alg., i/o in fft format, outputs an L-Tree (sec 3.2)
def ffLDL(G):
	d = len(G)
	if d==1:
		return (G,[])
	(G1,G2) = ffsplit(G)
	L = G2 / G1
	D1 = G1
	D2 =  G1 - L*conjugate(L)*G1 
	return (L, [ffLDL(D1),ffLDL(D2)] )

# ffLQ, i/o in fft format, outputs an L-Tree (sec 3.2)
def LDTree(f):
	F = fft(f)
	G = F*conjugate(F)
	T = ffLDL(G)
	return T

# ffBabai alg., i/o in base B, fft format 
def ffBabai_aux(T,t):
	if len(t)==1:
		return vector([round(t.real)])
	(t1,t2) = ffsplit(t)
	(L,[T1,T2]) = T
	z2 = ffBabai_aux(T2,t2)
	tb1 = t1 + (t2-z2) * conjugate(L)
	z1 = ffBabai_aux(T1,tb1)
	return ffmerge((z1,z2))

# ffBabai alg., i/o in canonical base, coef. format
def ffBabai(f,T,c):
	t = fft(c) / fft(f)
	z = ffBabai_aux(T,t)
	return ifft(z)

########################################################################
## Testing
########################################################################
# k = 10
# n = 2**k

# print "dimension : ", n 
# f = vector([random.randint(-10,10) for i in range(n)])
# print "base: ", f
# T = LDTree(f)
# print "LD-Tree Constructed "
# verifier = create_verifier(f)
# print "Babai Verifier constructed (explicit QDR computation)"



# print "Unit test ffBabai ",

# N = 10000
# for a in range(N):
# 	c = vector([random.uniform(-100,100) for i in range(n)])
# 	z = ffBabai(f,T,c)
# 	##verif_babai(verifier, 1.*c, z)
# 	if (100*a % N == 0):
# 		print 100*a / N,"%" ,
# 		sys.stdout.flush()

# print
# print " Passed !"