import numpy as np

def computeCost(X,y,theta):
	
	m = len(y) 
	J = np.sum(np.power(X.dot(theta) - y,2))
	J = (1.0/(m*2)) * J
	return J