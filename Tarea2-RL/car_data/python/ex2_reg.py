## Machine Learning Online Class - Exercise 2: Logistic Regression
## Initialization
import costFunctionReg
import predict
import plotData
import mapFeature
import plotDecisionBoundary
import numpy as np
from scipy.optimize import fmin_bfgs
from sklearn.preprocessing import PolynomialFeatures


def reg_log_reg(X,y):
## =========== Part 1: Regularized Logistic Regression ============
	X1 = mapFeature.mapFeature3(X[:,0], X[:,1],X[:,2], X[:,3],X[:,4], X[:,5])
	#print(X2.shape)
	#poly = PolynomialFeatures(degree=6)
	#X1 = poly.fit_transform(X)
	#print(X.shape,X1.shape)
	#X = mapFeature.mapFeature(X[:,0], X[:,1])
	m,n = X1.shape
	initial_theta = np.zeros((n, 1))

	# Set regularization parameter lambda to 1 (you should vary this)
	lambda_reg = 1.0

	# Set Options
	# Optimize
	myargs=(X1, y, lambda_reg)
	theta = fmin_bfgs(costFunctionReg.costFunctionReg, x0=initial_theta, args=myargs)
	print('theta:')
	print(theta)
	return theta
	# Compute accuracy on our training set
#p = predict.predict(theta, X);


#print('Train Accuracy: {:f}'.format(np.mean(p == y) * 100))
#raw_input('\nPress enter to finish\n')