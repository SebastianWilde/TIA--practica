import matplotlib.pyplot as plt

def plotData(x,y):
	plt.plot(x,y,'rx',markersize=10,label='Datos de entrenamiento')
	plt.show(block=False)