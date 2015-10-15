# ffo.py
# Public Domain
# Provided by Leo Ducas and Thomas Prest
# Reference implementation of algorithms proposed in
# [Fast Fourier Orthogonalization, Ducas and Prest]
# To be used for testing purposes only

from math import sqrt
from math import isnan as isNaN

from numpy.fft import rfft,irfft,fft,ifft
from numpy import inner, zeros,sign, conjugate
from numpy import roll,transpose,diag
from numpy.linalg import qr,inv
from numpy import matrix, vectorize
from numpy import array as vector
from math import log

import random


infinity = float('inf')
NaN = float('nan')


def log2(n):
	k = int(round(log(n)/log(2)))
	assert n == 2**k , "Non powers of 2 dimension : not implemented"
	return k

def bit_rev_order(k):
	d = 2**k
	br = [	int('{:0{width}b}'.format(i, width=k)[::-1],2) for i in range(d)]
	return br

def circulant_matrix(f):
	d = len(f)
	L = [f]
	for i in xrange(d-1):
		f = roll(f,1)
		L+= [f]
	return 1.*matrix(L)

def M_d1(f):
	d = len(f)
	k = log2(d)
	br = bit_rev_order(k)	
	M = circulant_matrix(f)
	P = matrix(d*[d*[0.]])
	for i in range(d):
		for j in range(d):
			P[br[i],br[j]]  = M[i,j]
	return P


def V_d1(f):
	d = len(f)
	k = log2(d)
	br = bit_rev_order(k)	
	P = zeros(d)
	for i in range(d):
		P[br[i]]  = f[i]
	return P


# def qdr(M):
# 	(d,_) = M.shape
# 	(Q,R) = qr(M)
# 	D = vector(diag(R).flat)
# 	for i in xrange(d):

# 		if D[i]!=0:
# 			x = 1/(D[i])
# 		else:
# 			x = 0
# 		R[i,:] *= x
# 	if (max([abs(x) for x in D])/min([abs(x) for x in D])) > 2.**30:
# 		print "Conditioning number too large"
# 		assert False

# 	return (Q,D,R)

# def ldq(M):
# 	(Q,D,R) = qdr(transpose(M))	
# 	return (transpose(R), D, transpose(Q))

##Dealing with non-idependant vector, 
##incorrect results for X with big conditionning number, \kappa(X) > ~2^30
def ldq(X):

	rows,cols=X.shape
	Y=zeros(cols)
	Q=zeros([rows,cols])
	D = rows*[0.]
	L = zeros([rows,rows])
	non_ind_c = 0

	D[0] = 1.*abs(sqrt(inner(X[0,:],X[0,:])))
	Q[0,:]= X[0,:] * (1./ D[0])
	for j in xrange(1,rows):
		Y=vector(X[j,:])
		for i in xrange(0,j):
			Y=Y- inner(X[j,:],Q[i,:])*Q[i,:]
		D[j] = sqrt(inner(Y,Y))
		if D[j]/D[0] < 2.**(-40): 
			non_ind_c += 1
			D[j] = infinity
		else:
			Q[j,:]= Y * (1./ D[j])
	for i in xrange(rows):
		L[i,i] = 1
		for j in xrange(i):
			L[i,j] = inner(X[i,:],Q[j,:]).flat[0] / D[j]

	print "Gram-Schmidt elimated ", non_ind_c , "vectors out of ", rows

	return (matrix(L),vector(D),matrix(Q))





def create_verifier(f):
	B =  M_d1(f)
	(L,D,Q) = ldq(B)
	return (B, D, Q)

def verif_babai(verifier, c, z):
	(B, D, Q) = verifier
	vc = V_d1(c)
	vz = V_d1(z)
	d = vz * B - vc
	qd = (d * transpose(Q)) / D
	for x in qd.flat:
		assert abs(x) <= .5 , "Babai Verification failed on target \n" +str(c)  + "\n and result \n" +str(z)



