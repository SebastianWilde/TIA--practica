# Machine Learning Online Class - Exercise 2: Logistic Regression
# Initialization
import numpy as np
from scipy.optimize import fmin
from scipy.optimize import fmin_bfgs


import plotData
import costFunction
import plotDecisionBoundary
import predict
import sigmoid
def reg_log(X,y):
	# ============ Part 2: Compute Cost and Gradient ============
	m,n = X.shape
	# Add intercept term to x and X_test
	X1 = np.column_stack((np.ones((m,1)),X))
	# Initialize fitting parameters
	initial_theta = np.zeros((n + 1, 1))
	# Compute and display initial cost and gradient
	cost, grad = costFunction.costFunction(initial_theta, X1, y,True)
	print('Cost at initial theta (zeros): {:f}'.format(cost))
	print('Gradient at initial theta (zeros):')
	print(grad)
	raw_input('\nProgram paused. Press enter to continue.\n')
	# ============= Part 3: Optimizing using fminunc  =============
	#  Set options for fminunc
	myargs=(X1, y)
	theta = fmin(costFunction.costFunction, x0=initial_theta, args=myargs)
	theta, cost, _, _, _, _, _ = fmin_bfgs(costFunction.costFunction, x0=theta, args=myargs, full_output=True)
	# Print theta to screen
	print('Cost at theta found by fmin: {:f}'.format(cost))
	print('theta:')
	print(theta)
	return theta
	# ============== Part 4: Predict and Accuracies ==============
	#  After learning the parameters, you'll like to use it to predict the outcomes
	#  on unseen data. In this part, you will use the logistic regression model
	#  to predict the probability that a student with score 45 on exam 1 and 
	#  score 85 on exam 2 will be admitted.
	#
	#  Furthermore, you will compute the training and test set accuracies of 
	#  our model.
	#
	#  Your task is to complete the code in predict.m

	#  Predict probability for a student with score 45 on exam 1 
	#  and score 85 on exam 2 
"""
	prob = sigmoid.sigmoid(np.dot(np.array([1, 45, 85]), theta))
	print('For a student with scores 45 and 85, we predict an admission probability of {:f}'.format(prob))


	# Compute accuracy on our training set
	p = predict.predict(theta, X1);

	print('Train Accuracy: {:f}'.format(np.mean(p == y) * 100))

	raw_input('\nProgram paused. Press enter to continue.\n')
	"""