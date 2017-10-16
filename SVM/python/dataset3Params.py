import numpy as np
import svmTrain as svmt
import gaussianKernelGramMatrix as gkgm
def dataset3Params(X, y, Xval, yval):
#EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
#where you select the optimal (C, sigma) learning parameters to use for SVM
#with RBF kernel
#   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
#   sigma. You should complete this function to return the optimal C and 
#   sigma based on a cross-validation set.
#

# You need to return the following variables correctly.
	C = 1
	sigma = 0.3

# ====================== YOUR CODE HERE ======================
# Instructions: Fill in this function to return the optimal C and sigma
#               learning parameters found using the cross validation set.
#               You can use svmPredict to predict the labels on the cross
#               validation set. For example, 
#                   predictions = svmPredict(model, Xval);
#               will return the predictionns on the cross validation set.
#
#  Note: You can compute the prediction error using 
#        mean(double(predictions ~= yval))
#
	#Algunos posibles valores para C y sigma
	posibles_valores = [0.01,0.03,0.1,0.3,1,3,10,30]
	#Promedio de predicciones malas
	error_inicial = 1.0

	for nuevo_C in posibles_valores:
		for nuevo_sigma in posibles_valores:
			model = svmt.svmTrain(X, y, C, "gaussian", sigma=sigma)
			prediccion = model.predict(gkgm.gaussianKernelGramMatrix(Xval, X))
			nuevo_error = np.mean((prediccion != yval).astype(int))
			if (error_inicial > nuevo_error): 
				error_inicial = nuevo_error
				C = nuevo_C
				sigma = nuevo_sigma
	return C, sigma