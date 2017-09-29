###Transformar la informacion de car-data
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

##Prueba con car-prueba test
test = np.genfromtxt('car-prueba.data',delimiter=",",dtype='str')
fila_test,col_test = test.shape
new_test = np.zeros((test.shape))
		#Asignando valores numericos al data set car-prueba
for i in xrange(0,col_test):
	for j in xrange(0,fila_test):
		new_test[j,i] = metadata[i][str(test[j,i])]

np.savetxt('new_car.txt', new_data,fmt='%i',delimiter=",")
np.savetxt('new_car_prueba.txt', new_test,fmt='%i',delimiter=",")