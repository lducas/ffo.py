# ffo.py
# Public Domain
# Provided by Leo Ducas and Thomas Prest
# Reference implementation of algorithms proposed in
# [Fast Fourier Orthogonalization, Ducas and Prest]
# To be used for testing purposes only


import sys
import random
from numpy import array
from verif import create_verifier, verif_babai
from ffo_NaN import LDTree, ffBabai
from cyclo import embedding


k = 6
n = 2**k

print
print "testing ffo in the convolution ring R[x]/(x^64-1)"
print

x = [random.randint(-10,10) for i in range(n)]
f = array(x)
print "base: ", f
T = LDTree(f)
print "LD-Tree Constructed "
verifier = create_verifier(f)
print "Babai Verifier constructed (explicit QDR computation)"

print "Unit test ffBabai (1000 trials) ",

N = 1000
for a in range(1,N+1):
	c = array([random.uniform(-100,100) for i in range(n)])
	z = ffBabai(f,T,c)
	verif_babai(verifier, c, z)
	if ((10*a) % N == 0):
		print 100*a / N,"%" ,
		sys.stdout.flush()

print
print " Passed !"

print
print "testing ffo in the convolution ring R[x]/(x^32+1)"
print

x = [random.randint(-10,10) for i in range(n/2)]
f = embedding(array(x))
print "base: ", f
T = LDTree(f)
print "LD-Tree Constructed "
verifier = create_verifier(f)
print "Babai Verifier constructed (explicit QDR computation)"

print "Unit test ffBabai (1000 trials) ",

N = 1000
for a in range(1,N+1):
	c = array([random.uniform(-100,100) for i in range(n/2)])
	z = ffBabai(f,T,embedding(c))
	verif_babai(verifier, embedding(c), z)
	if ((10*a) % N == 0):
		print 100*a / N,"%" ,
		sys.stdout.flush()

print
print " Passed !"