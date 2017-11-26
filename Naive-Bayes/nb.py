###Transformar la informacion de car-data
# buying = {'vhigh':1, 'high':2, 'med':3, 'low':4}
# maint = {'vhigh':1, 'high':2, 'med':3, 'low':4}
# doors = {'2':1, '3':2, '4':3, '5more':4}
# persons = {'2':1, '4':2, 'more':3}
# lug_boot = {'small':1, 'med':2, 'big':3}
# safety ={ 'low':1, 'med':2, 'high':3}
# classes = {'unacc':1,'acc':2 ,'good':3,'vgood':4}
#metadata = [buying,maint,doors,persons,lug_boot,safety,classes]
def FindMaxDictionary(dict):
    mayor = {}
    mayor_num = 0
    for key in dict.items():
        if key[1] > mayor_num:
            mayor_num = key[1]
            mayor = {key[0]:key[1]}
    return mayor


import pandas as pd
import numpy as np
metadata = ['buying','maint','doors','persons','lug_boot','safety','class']


#Entrenamiento

data = pd.read_csv('car.data',names = metadata)
#Datos clases
datos_clases = {}
clases = data['class'].value_counts().index.tolist()
#print(clases)
probabilidades_clases = (data['class'].value_counts() / len(data['class'])).tolist()

for i in range(len(clases)):
    datos_clases[clases[i]] = probabilidades_clases[i]
#print(datos_clases)

#Trabajando con las caracteristicas, look_up
n_caracterisiticas = len(metadata) - 1
look_up = {}
for j in range(n_caracterisiticas):
    propabilidad_parcial = pd.crosstab(index=data[metadata[j]],columns=data['class']).apply(lambda r:r/r.sum(),axis=0)
    probabilidad_total = pd.value_counts(data[metadata[j]])
    look_up.update({metadata[j]:[propabilidad_parcial,probabilidad_total]})
#print(look_up)
#Testing
data_test = pd.read_csv('car-prueba.data',names = metadata)
#print(data_test)
n_entradas = data_test.shape[0]
clasificacion = []
for datos in data_test.values:
    #print(datos)
    entradas = datos[:n_caracterisiticas]
    resultados_clases = {}
    for clase in clases:
        P_x_dado_y = 1.0
        P_x = 1.0
        for h in range(n_caracterisiticas):
            P_x_dado_y *= look_up[metadata[h]][0][clase][entradas[h]]
            P_x *= look_up[metadata[h]][1][entradas[h]]
        P_x_dado_y *=  datos_clases[clase]
        resultados_clases.update({clase:P_x_dado_y/P_x})
    clasificado = FindMaxDictionary(resultados_clases)
    #print("clasificado",clasificado)
    clasificacion.append(clasificado)
    #print("Clasificacion",clasificacion)
#print("\n",clasificacion)
#Resultados
y = data_test['class'].tolist()
resultado = 0.0
for item,original in zip(clasificacion,y):
    result_clasificacion = item.popitem()
    print("Resultado de la clasificacion",result_clasificacion[0])
    print("Probabilidad",result_clasificacion[1])
    print("Resultado real",original)
    if (result_clasificacion[0] ==original ):
        resultado+=1.0

print("Porcentaje de acierto",resultado/n_entradas*100)









