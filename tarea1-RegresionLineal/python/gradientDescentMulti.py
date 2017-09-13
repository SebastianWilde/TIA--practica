
import numpy as np
import computeCostMulti as ccm
def gradientDescentMulti(X, y, theta, alpha, num_iters):
	m = len(y)
	J_history = np.zeros((num_iters,1))
	for i in xrange(num_iters):
		theta = theta - alpha*(1.0/m) * np.transpose(X).dot(X.dot(theta) - np.transpose([y]))
		J_history[i] = ccm.computeCostMulti(X,y,theta)
	return theta,J_history