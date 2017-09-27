
#Clases del data set
buying = {'vhigh':1, 'high':2, 'med':3, 'low':4}
maint = {'vhigh':1, 'high':2, 'med':3, 'low':4}
doors = {'2':1, '3':2, '4':3, '5more':4}
persons = {'2':1, '4':2, 'more':3}
lug_boot = {'small':1, 'med':2, 'big':3}
safety ={ 'low':1, 'med':2, 'high':3}
clase = {'unacc':1,'acc':2 ,'good':3,'vgood':4}
metadata = [buying,maint,doors,persons,lug_boot,safety,clase]
import numpy as np

data = np.genfromtxt('car.data',delimiter=",",dtype='str')
fila,col = data.shape
new_data = np.zeros((data.shape))


#Asignando valores numericos al data set car
for i in xrange(0,col):
	for j in xrange(0,fila):
		new_data[j,i] = metadata[i][str(data[j,i])]


#Captura la data para las 6 clases
X = new_data[:,:6]
y_unacc = y_acc = y_good = y_vgood = new_data[:,6]

#print(new_data.shape,X.shape,y_unacc.shape)
##Para el caso unacc
y_unacc[np.where(y_unacc!=1)] = 0
##Para acc
y_acc[np.where(y_acc==2)] = 1
y_acc[np.where(y_acc!=2)] = 0
##Para good
y_good[np.where(y_good==3)] = 1
y_good[np.where(y_good!=3)] = 0
##Para vgood
y_vgood[np.where(y_vgood==4)] = 1
y_vgood[np.where(y_vgood!=4)] = 0

##Regresion logistica para cada clase
import ex2
print("\n---------------Iniciando con la clase unacc---------------\n")
theta_unacc = ex2.reg_log(X,y_unacc)
print("\n---------------Iniciando con la clase acc---------------\n")
theta_acc = ex2.reg_log(X,y_acc)
print("\n---------------Iniciando con la clase good---------------\n")
theta_good = ex2.reg_log(X,y_good)
print("\n---------------Iniciando con la clase vgood---------------\n")
theta_vgood = ex2.reg_log(X,y_vgood)

##Prueba con car-prueba test
test = np.genfromtxt('car-prueba.data',delimiter=",",dtype='str')
fila_test,col_test = test.shape
new_test = np.zeros((test.shape))


#Asignando valores numericos al data set car-prueba
for i in xrange(0,col_test):
	for j in xrange(0,fila_test):
		new_test[j,i] = metadata[i][str(test[j,i])]
print (test,new_test)
#Captura la data para las 6 clases
X_test = new_test[:,:6]
# Add intercept term to x and X_test
m = X_test.shape[0]
X1 = np.column_stack((np.ones((m,1)),X_test))
y_unacc_test = y_acc_test = y_good_test = y_vgood_test = new_test[:,6]
##Para el caso unacc
y_unacc_test[np.where(y_unacc_test!=1)] = 0
##Para acc
y_acc_test[np.where(y_acc_test==2)] = 1
y_acc_test[np.where(y_acc_test!=2)] = 0
##Para good
y_good_test[np.where(y_good_test==3)] = 1
y_good_test[np.where(y_good_test!=3)] = 0
##Para vgood
y_vgood_test[np.where(y_vgood_test==4)] = 1
y_vgood_test[np.where(y_vgood_test!=4)] = 0

import predict

p = predict.predict(theta_unacc, X1);
print('Train Accuracy to class unacc: {:f}'.format(np.mean(p == y_unacc_test) * 100))

p = predict.predict(theta_acc, X1);
print('Train Accuracy to class acc: {:f}'.format(np.mean(p == y_acc_test) * 100))

p = predict.predict(theta_good, X1);
print('Train Accuracy to class good: {:f}'.format(np.mean(p == y_good_test) * 100))

p = predict.predict(theta_vgood, X1);
print('Train Accuracy to class vgood: {:f}'.format(np.mean(p == y_vgood_test) * 100))
