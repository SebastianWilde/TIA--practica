import numpy as np
from sigmoid import sigmoid
def costFunctionReg(theta, X, y, lambda_reg,return_grad = False):
#COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
#   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
#   theta as the parameter for regularized logistic regression and the
#   gradient of the cost w.r.t. to the parameters. 

# Initialize some useful values
	m = len(y) # number of training examples

# You need to return the following variables correctly 
	J = 0
	grad = np.zeros(theta.shape);

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta

	primer_elemento = np.dot(y.T,sigmoid(np.dot(X,theta)))
	segundo_elemento = np.dot((1-y).T,sigmoid(np.dot(X,theta)))
	regularizacion = (float(lambda_reg)/(2*m)) * np.power(theta[1:theta.shape[0]],2).sum()
	J = (-1./m) * (primer_elemento+segundo_elemento).sum()+regularizacion

	#J = (-1./m) * (np.dot(y.T,log(sigmoid(np.dot(X,theta)))) + np.dot((1-y).T*log(1-sigmoid(np.dot(X,theta))))).sum() + 
	#(float(lambda_reg)/(2*m))*np.power(theta[1:theta.shape[0]],2).sum()

# =============================================================
	if return_grad == True:
		regu_theta = theta[1:theta.shape[0]]
		regu_theta = regu_theta.insert(0,0)
		grad = (1./m) * (np.dot(((sigmoid(np.dot(X*theta)) - y).T * X)) + np.dot(lambda_reg*regu_theta.T))
		return J,grad
	else:
		return J
