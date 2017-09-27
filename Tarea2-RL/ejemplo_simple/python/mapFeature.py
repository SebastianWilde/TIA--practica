import numpy as np
from pandas import Series
def mapFeature(X1, X2):
# MAPFEATURE Feature mapping function to polynomial features
#
#   MAPFEATURE(X1, X2) maps the two input features
#   to quadratic features used in the regularization exercise.
#
#   Returns a new feature array with more features, comprising of 
#   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
#   for a total of 1 + 2 + ... + (degree+1) = ((degree+1) * (degree+2)) / 2 columns
#
#   Inputs X1, X2 must be the same size
	degree = 6
	#quads = Series([X1**(i-j) * X2**j for i in range(1,degree+1) for j in range(i+1)])
	#return  Series([1]).append([Series(X1), Series(X2), quads])
	out = np.ones(( X1.shape[0], sum(range(degree + 2)) ))
	curr_column = 1
	for i in xrange(1, degree + 1):
		for j in xrange(i+1):
			out[:,curr_column] = np.power(X1,i-j) * np.power(X2,j)
			curr_column += 1
	return out



def mapFeature2(X1, X2):
	degree = 6
	quads = Series([X1**(i-j) * X2**j for i in range(1,degree+1) for j in range(i+1)])
	return  Series([1]).append([Series(X1), Series(X2), quads])