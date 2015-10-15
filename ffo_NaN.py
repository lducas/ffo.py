# ffo.py
# Public Domain
# Provided by Leo Ducas and Thomas Prest
# Reference implementation of algorithms proposed in
# [Fast Fourier Orthogonalization, Ducas and Prest]
# To be used for testing purposes only

from math import sqrt
from numpy.fft import fft,ifft
from numpy import zeros, conjugate,exp,pi
from numpy import array,vectorize

def cheated_inverse(x):
	if abs(x)<2.**(-30):
		return 0.j
	return 1./x


def ffmerge((F1,F2)):
	d = 2*len(F1)
	F = 0.j*zeros(d)
	w = exp(-2.j*pi / d)
	W = array([w**i for i in range(d/2)])
	F[:d/2] = F1 + W* F2
	F[d/2:] = F1 - W* F2
	return F


def ffsplit(F):
	d = len(F)
	winv = exp(2.j*pi / d)
	Winv = array([winv**i for i in range(d/2)])
	F1 = .5* (F[:d/2] + F[d/2:])
	F2 = .5* (F[:d/2] - F[d/2:]) * Winv
	return (F1,F2)


def ffLDL(G):
	d = len(G)
	if d==1:
		return (G,[])
	(G1,G2) = ffsplit(G)
	L = G2 * vectorize(cheated_inverse)(G1)
	D1 = G1
	D2 =  G1 - L*conjugate(L)*G1 
	return (L, [ffLDL(D1),ffLDL(D2)] )


def LDTree(f):
	F = fft(f)
	G = F*conjugate(F)
	T = ffLDL(G)
	return T


def ffBabai_aux(T,t):
	if len(t)==1:
		if abs(T[0]) <2.**(-30):
			return array([0])
		return array([round(t.real)])
	(t1,t2) = ffsplit(t)
	(L,[T1,T2]) = T
	z2 = ffBabai_aux(T2,t2)
	tb1 = t1 + (t2-z2) * conjugate(L)
	z1 = ffBabai_aux(T1,tb1)
	return ffmerge((z1,z2))


def ffBabai(f,T,c):
	t = fft(c) * vectorize(cheated_inverse)(fft(f))
	z = ffBabai_aux(T,t)
	return vectorize(round)(ifft(z).real)

