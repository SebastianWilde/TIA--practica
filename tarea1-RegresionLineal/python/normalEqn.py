import numpy as np


def normalEqn(X, y):
	theta = np.zeros((X.shape[1],1))
	theta = np.linalg.pinv(np.transpose(X).dot(X)).dot(np.transpose(X)).dot(y)
	return theta	