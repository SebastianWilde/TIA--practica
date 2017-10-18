## Machine Learning Online Class
#  Exercise 6 | Support Vector Machines
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     gaussianKernel.m
#     dataset3Params.m
#     processEmail.m
#     emailFeatures.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization
## =============== Part 1: Loading and Visualizing Data ================
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#
import scipy.io as sci
import numpy as np
import plotData as pd
import svmTrain as svmt
import visualizeBoundaryLinear as vbl
import gaussianKernel as gk
import visualizeBoundary as vb
import dataset3Params as dp
def svm_model(X,y,Xval,yval):
	C, sigma = dp.dataset3Params(X, y, Xval, yval)
	# Train the SVM
	model = svmt.svmTrain(X, y, C, "gaussian", sigma=sigma)
	return C,sigma,model