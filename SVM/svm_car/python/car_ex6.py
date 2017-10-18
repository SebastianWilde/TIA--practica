####Regresion logistica con car data sin regularizacion--------------------
#Clases del data set
import numpy as np
from sigmoid import sigmoid


data = np.loadtxt('new_car.txt',delimiter=",")
fila,col = data.shape
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


data_test = np.loadtxt('new_car_prueba.txt',delimiter=",")
fila_test,col_test = data_test.shape

#Captura la data para las 6 clases
X_test = data_test[:,:6]
# Add intercept term to x and X_test
m = X_test.shape[0]
X1 = np.column_stack((np.ones((m,1)),X_test))
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

##Calculando los modelos para cada clase
import ex6
print("\n---------------Iniciando con la clase unacc---------------\n")
C_unacc,sigma_unacc,model_unacc = ex6.svm_model(X,y_unacc,X_test,y_unacc_test)
print("\n Para unacc C es %f y sigma es %f \n" %(C_unacc,sigma_unacc))

print("\n---------------Iniciando con la clase acc---------------\n")
C_acc,sigma_acc,model_acc = ex6.svm_model(X,y_acc,X_test,y_acc_test)
print("\n Para acc C es %f y sigma es %f \n" %(C_acc,sigma_acc))

print("\n---------------Iniciando con la clase unacc---------------\n")
C_good,sigma_good,model_good = ex6.svm_model(X,y_good,X_test,y_good_test)
print("\n Para good C es %f y sigma es %f \n" %(C_good,sigma_good))

print("\n---------------Iniciando con la clase vgood---------------\n")
C_vgood,sigma_vgood,model_vgood = ex6.svm_model(X,y_vgood,X_test,y_vgood_test)
print("\n Para vgood C es %f y sigma es %f \n" %(C_vgood,sigma_vgood))

##---------------Prueba con car-prueba test------------------------
print("\n-------Iniciando la prueba con car-prueba----------------\n")


p1 = model_unacc.predict(gkgm.gaussianKernelGramMatrix(X_test, X))
print('\nTrain Accuracy to class unacc: {:f}'.format(np.mean(p1 == y_unacc_test) * 100))
p2 = model_acc.predict(gkgm.gaussianKernelGramMatrix(X_test, X))
print('Train Accuracy to class acc: {:f}'.format(np.mean(p2 == y_acc_test) * 100))
p3 = model_good.predict(gkgm.gaussianKernelGramMatrix(X_test, X))
print('Train Accuracy to class good: {:f}'.format(np.mean(p3 == y_good_test) * 100))
p4 = model_vgood.predict(gkgm.gaussianKernelGramMatrix(X_test, X))
print('Train Accuracy to class vgood: {:f}'.format(np.mean(p4 == y_vgood_test) * 100))
