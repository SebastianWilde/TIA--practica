## Machine Learning Online Class - Exercise 2: Logistic Regression
## Initialization
import costFunctionReg
import predict
import plotData
import mapFeature
import plotDecisionBoundary
import numpy as np
from scipy.optimize import fmin_bfgs


def reg_log_reg(X,y):
## =========== Part 1: Regularized Logistic Regression ============
	X = mapFeature.mapFeature(X[:,0], X[:,1])
	m,n = X.shape
	# Initialize fitting parameters
	initial_theta = np.zeros((n,1))
	# Set regularization parameter lambda to 1
	lambda_reg = 1.0

	# Compute and display initial cost and gradient for regularized logistic
	# regression
	cost = costFunctionReg.costFunctionReg(initial_theta, X, y, lambda_reg)

	print('Cost at initial theta (zeros): {:f}'.format(cost))
	## ============= Part 2: Regularization and Accuracies =============
	# Initialize fitting parameters
	initial_theta = np.zeros((n, 1))

	# Set regularization parameter lambda to 1 (you should vary this)
	lambda_reg = 1.0

	# Set Options
	# Optimize
	myargs=(X, y, lambda_reg)
	theta = fmin_bfgs(costFunctionReg.costFunctionReg, x0=initial_theta, args=myargs)
	return theta
	# Compute accuracy on our training set
#p = predict.predict(theta, X);


#print('Train Accuracy: {:f}'.format(np.mean(p == y) * 100))
#raw_input('\nPress enter to finish\n')