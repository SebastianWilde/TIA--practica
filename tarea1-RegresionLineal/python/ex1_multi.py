
import numpy as np 
import matplotlib.pyplot as plt
import featureNormalize as fn
import gradientDescentMulti as gdm
import normalEqn as ne

## Initialization

## ================ Part 1: Feature Normalization ================


print('Cargando datos ...\n');
data = np.loadtxt('ex1data2.txt',delimiter=",")

X = data[:,:2]
y = data[:,2]
m = len(y)

# Print out some data points
print('First 10 examples from the dataset: \n')
for i in xrange(10):
	print ("x = [{:.0f} {:.0f}], y = {:.0f}".format(X[i,0], X[i,1], y[i]))

raw_input('Program paused. Press enter to continue.\n')

# Scale features and set them to zero mean
print('Normalizing Features ...\n')

X_norm, mu, sigma = fn.featureNormalize(X)

# Add intercept term to X
X1 = np.column_stack((np.ones((m,1)),X_norm))


# ================ Part 2: Gradient Descent ================

print('Running gradient descent ...\n');

# Choose some alpha value
alfa = 0.01
num_iters = 400

# Init Theta and Run Gradient Descent 
theta = np.zeros((3, 1))
theta, J_history = gdm.gradientDescentMulti(X1, y, theta, alfa, num_iters)

# Plot the convergence graph
plt.plot(xrange(J_history.size), J_history, "-b", linewidth=2 )
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show(block=False)

# Display gradient descent's result
print("{:f}, {:f}, {:f}".format(theta[0,0], theta[1,0], theta[2,0]))
print("")

# Estimate the price of a 1650 sq-ft, 3 br house
area_normalizada = (1650 - float(mu[:,0]))/float(sigma[:,0])
br_normalizada = (3 - float(mu[:,1]))/float(sigma[:,1])
new_house = np.array([1,area_normalizada,br_normalizada])
price = new_house.dot(theta)

# ============================================================

print("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n ${:,.2f}".format(price[0]))

raw_input('Program paused. Press enter to continue.\n')


# ================ Part 3: Normal Equations ================

print('Solving with normal equations...\n')

# Load Data
data = np.loadtxt('ex1data2.txt', delimiter=",")
X = data[:,:2]
y = data[:,2]
m = len(y) # number of training examples


# Add intercept term to X
X1 = np.column_stack((np.ones((m,1)),X))

# Calculate the parameters from the normal equation
theta = ne.normalEqn(X1, y)

# Display normal equation's result
print('Theta computed from the normal equations:')
print("{:f}, {:f}, {:f}".format(theta[0], theta[1], theta[2]))
print('')


# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
price = np.array([1, 1650, 3]).dot(theta)


# ============================================================

print("Predicted price of a 1650 sq-ft, 3 br house (using normal equiations):\n ${:,.2f}".format(price))
