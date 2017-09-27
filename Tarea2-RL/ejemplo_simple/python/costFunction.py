
import numpy as np
from sigmoid import sigmoid
def costFunction(theta, X, y,return_grad = False):
#COSTFUNCTION Compute cost and gradient for logistic regression
#   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
#   parameter for logistic regression and the gradient of the cost
#   w.r.t. to the parameters.

# Initialize some useful values
	m = len(y) # number of training examples

# You need to return the following variables correctly 
	J = 0;
	grad = np.zeros(theta.shape)

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta
#
# Note: grad should have the same dimensions as theta
#
#	print ("X shape",X.shape)
#	print ("theta shape",theta.shape)
#	print ("y shape",y.shape)
	J = (-1./m) * np.sum(y.T*np.log(sigmoid(X.dot(theta))) + np.transpose(1-y) * np.log(1-sigmoid(X.dot(theta))))	
 	grad = (1./m) * np.dot(sigmoid( np.dot(X,theta) ).T - y, X).T
	# =============================================================
	
	if return_grad == True:
		return J,grad
	else:
		return J
