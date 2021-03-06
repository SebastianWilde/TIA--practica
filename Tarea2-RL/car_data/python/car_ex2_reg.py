####Regresion logistica con car data con regularizacion--------------------
#Clases del data set
import numpy as np
import mapFeature
from sklearn.preprocessing import PolynomialFeatures

data = np.loadtxt('new_car.txt',delimiter=",")
fila,col = data.shape
print (data[238,6],data[239,6])
#Captura la data para las 6 clases
X = data[:,:6]
y_unacc = np.zeros((fila))
y_acc = np.zeros((fila))
y_good = np.zeros((fila))
y_vgood = np.zeros((fila))

for i in xrange (0,fila):
	if (data[i,6] == 1):
		y_unacc[i] = 1

for i in xrange (0,fila):
	if (data[i,6] == 2):
		y_acc[i] = 1

for i in xrange (0,fila):
	if (data[i,6] == 3):
		y_good[i] = 1


for i in xrange (0,fila):
	if (data[i,6] == 4):
		y_vgood[i] = 1

##Regresion logistica para cada clase
import ex2_reg
print("\n---------------Iniciando con la clase unacc---------------\n")
theta_unacc = ex2_reg.reg_log_reg(X,y_unacc)
print("\n---------------Iniciando con la clase acc---------------\n")
theta_acc = ex2_reg.reg_log_reg(X,y_acc)
print("\n---------------Iniciando con la clase good---------------\n")
theta_good = ex2_reg.reg_log_reg(X,y_good)
print("\n---------------Iniciando con la clase vgood---------------\n")
theta_vgood = ex2_reg.reg_log_reg(X,y_vgood)



##Prueba con car-prueba test
##---------------Prueba con car-prueba test------------------------
print("\n-------Iniciando la prueba con car-prueba----------------\n")
data_test = np.loadtxt('new_car_prueba.txt',delimiter=",")
fila_test,col_test = data_test.shape

#Captura la data para las 6 clases
X_test = data_test[:,:6]
# Add intercept term to x and X_test
#poly = PolynomialFeatures(degree=4)
#X1 = poly.fit_transform(X_test)
X1 = mapFeature.mapFeature3(X_test[:,0], X_test[:,1],X_test[:,2], 
	X_test[:,3],X_test[:,4], X_test[:,5])

y_unacc_test = np.zeros((fila_test))
y_acc_test = np.zeros((fila_test))
y_good_test = np.zeros((fila_test))
y_vgood_test = np.zeros((fila_test))

for i in xrange (0,fila_test):
	if (data_test[i,6] == 1):
		y_unacc_test[i] = 1
for i in xrange (0,fila_test):
	if (data_test[i,6] == 2):
		y_acc_test[i] = 1
for i in xrange (0,fila_test):
	if (data_test[i,6] == 3):
		y_good_test[i] = 1

for i in xrange (0,fila_test):
	if (data_test[i,6] == 4):
		y_vgood_test[i] = 1


import predict

p1 = predict.predict(theta_unacc, X1)
print('Train Accuracy to class unacc: {:f}'.format(np.mean(p1 == y_unacc_test) * 100))

p2 = predict.predict(theta_acc, X1)
print('Train Accuracy to class acc: {:f}'.format(np.mean(p2 == y_acc_test) * 100))

p3 = predict.predict(theta_good, X1)
print('Train Accuracy to class good: {:f}'.format(np.mean(p3 == y_good_test) * 100))

p4 = predict.predict(theta_vgood, X1)
print('Train Accuracy to class vgood: {:f}'.format(np.mean(p4 == y_vgood_test) * 100))
