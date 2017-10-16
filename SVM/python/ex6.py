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

print('Loading and Visualizing Data ...\n')

# Load from ex6data1: 
# You will have X, y in your environment
data = sci.loadmat('ex6data1.mat')
X = data["X"]
y = data["y"]
# Plot training data
pd.plotData(X, y)

raw_input('Program paused. Press enter to continue.\n')

## ==================== Part 2: Training Linear SVM ====================
#  The following code will train a linear SVM on the dataset and plot the
#  decision boundary learned.
#

# Load from ex6data1: 
# You will have X, y in your environment
data = sci.loadmat('ex6data1.mat')
X = data["X"]
y = data["y"]

print('\nTraining Linear SVM ...\n')

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)
C = 1
model = svmt.svmTrain(X, y, C, "linear", 1e-3, 20)
#plt.close()
vbl.visualizeBoundaryLinear(X, y, model)

raw_input('Program paused. Press enter to continue.\n');

## =============== Part 3: Implementing Gaussian Kernel ===============
#  You will now implement the Gaussian kernel to use
#  with the SVM. You should complete the code in gaussianKernel.m
#
print('\nEvaluating the s ...\n')

x1 = np.array([1, 2 ,1])
x2 = np.array([0, 4 ,-1])
sigma = 2;
sim = gk.gaussianKernel(x1, x2, sigma)


print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = 2 :' \
'\n\t{:f}\n(this value should be about 0.324652)\n'.format(sim))

raw_input('Program paused. Press enter to continue.\n')

## =============== Part 4: Visualizing Dataset 2 ================
#  The following code will load the next dataset into your environment and 
#  plot the data. 
#

print('Loading and Visualizing Data ...\n')

# Load from ex6data2: 
# You will have X, y in your environment
data2 = sci.loadmat('ex6data2.mat')
X = data2["X"]
y = data2["y"]

# Plot training data
#plt.close()
pd.plotData(X, y)

raw_input('Program paused. Press enter to continue.\n')

## ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
#  After you have implemented the kernel, we can now use it to train the 
#  SVM classifier.
# 
print('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n')

# Load from ex6data2: 
# You will have X, y in your environment
data2 = sci.loadmat('ex6data2.mat')
X = data2["X"]
y = data2["y"]

# SVM Parameters
C = 1
sigma = 0.1

# We set the tolerance and max_passes lower here so that the code will run
# faster. However, in practice, you will want to run the training to
# convergence.
model= svmt.svmTrain(X, y, C, "gaussian")
#plt.close()
#vb.visualizeBoundary(X, y, model)

raw_input('Program paused. Press enter to continue.')
## =============== Part 6: Visualizing Dataset 3 ================
#  The following code will load the next dataset into your environment and 
#  plot the data. 
#

print('Loading and Visualizing Data ...\n')

# Load from ex6data3: 
# You will have X, y in your environment
data3 = sci.loadmat('ex6data3.mat')
X = data3["X"]
y = data3["y"]

# Plot training data
#plt.close()
pd.plotData(X, y)

raw_input('Program paused. Press enter to continue.')

## ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

#  This is a different dataset that you can use to experiment with. Try
#  different values of C and sigma here.
# 

# Load from ex6data3: 
# You will have X, y in your environment
data3 = sci.loadmat('ex6data3.mat')
X = data3["X"]
y = data3["y"]
Xval = data3["Xval"]
yval = data3["yval"]
# Try different SVM Parameters here
C, sigma = dp.dataset3Params(X, y, Xval, yval)

# Train the SVM
model = svmt.svmTrain(X, y, C, "gaussian", sigma=sigma)
print (C,sigma)
#plt.close()
#vb.visualizeBoundary(X, y, model)
print('Program paused. Press enter to continue.\n')