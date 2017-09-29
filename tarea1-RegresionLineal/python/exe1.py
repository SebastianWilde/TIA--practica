import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
import warmUpExercise as wue
import plotData as pd
import computeCost as cc
import gradientDescent as gd


##Parte 1: Funcion Basica------------------
#Funcion warmUpExercise.pyplot
print('Ejecutando warmUpExercise')
print('Matriz identidad 5x5')
print(wue.warmUpExercise())

raw_input('En pausa. Enter para continuar.\n')

##Parte 2: Plotting
print('Plotting Data')

data = np.loadtxt('ex1data1.txt',delimiter=",")
X = data[:,0]
y = data[:,1]
m = len(y) #numero de ejemplos a entrenar
pd.plotData(X,y)
raw_input('En pausa. Enter para continuar.\n')

##Parte 3: Gradiente descendiente
print('Gradiente descendiente...')
X1 = np.column_stack((np.ones((m,1)),X))
theta = np.zeros((2,1))
iteraciones = 1500
alfa = 0.01

#Funcion de costo
print cc.computeCost(X1,y,theta)

#Gradiente descendiente
theta = gd.gradientDescent(X1,y,theta,alfa,iteraciones)
print('Valores de theta encontrados por la gradiente descendiente')
print("{:f},{:f}".format(theta[0,0],theta[1,0]))

#Plot linear fit

plt.plot(X,X1.dot(theta),'-',label = 'Regresion lineal')
plt.legend(loc = 'lower right')
plt.draw()
#plt.hold(False)

#Prediccion para una poblacion de 35000 y 70000
prediccion1 = np.array([1,3.5]).dot(theta)
print ("Para una poblacion = 35000, la prediccion es {:f}".format(float(prediccion1*10000)))

prediccion2 = np.array([1,7]).dot(theta)
print ("Para una poblacion = 700000, la prediccion es {:f}".format(float(prediccion1*10000)))

raw_input('En pausa, presiona enter \n')

#Parte 4: Visualizacion J(theta0,theta1)

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i in xrange(len(theta0_vals)):
    for j in xrange(len(theta1_vals)):
        t = [[theta0_vals[i]], [theta1_vals[j]]]
J_vals[i,j] = cc.computeCost(X1, y, t)
J_vals = np.transpose(J_vals)

fig = plt.figure()
#ax = Axes3D(fig)
ax = fig.gca(projection='3d')
theta0_vals,theta1_vals = np.meshgrid(theta0_vals,theta1_vals)
surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap=cm.coolwarm, rstride=2, cstride=2)
fig.colorbar(surf)
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.show(block=False)
#plt.hold(False)


fig = plt.figure()
ax = fig.add_subplot(111)
cset = plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20), cmap=cm.coolwarm)
fig.colorbar(cset)
plt.xlabel('theta_0')
plt.ylabel('theta_1')
#plt.hold(True)
plt.plot(theta[0,0], theta[1,0], 'rx', markersize=10, linewidth=2)
plt.show(block=False)
#plt.hold(False)


raw_input('Program paused. Press enter to finish.\n')