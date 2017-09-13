import numpy as np
import computeCost as cc

def gradientDescent(X,y,theta,alfa,iterations):
	m = len(y)
	J_history = np.zeros((iterations,1))

	for i in xrange(iterations):
		temp0 = theta[0] - alfa*(1.0/m)*np.sum((theta[0]+ theta[1]*X[:,1]) - y)
		temp1 = theta[1] - alfa*(1.0/m)*np.sum(((theta[0]+ theta[1]*X[:,1]) - y) * X[:,1])
		theta[0] = temp0
		theta[1] = temp1
		J_history[i] = cc.computeCost(X,y,theta)

	return theta