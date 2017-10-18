import matplotlib.pyplot as plt
import plotData as pd
import numpy as np
import gaussianKernelGramMatrix as gkgm

def visualizeBoundary(X, y, model, varargin=0):
    """Plots a non-linear decision boundary learned by the SVM"""
    
    plt.figure(figsize=(12,8))

    x1plot = np.linspace(X[:,0].min(), X[:,0].max(), 100).T
    x2plot = np.linspace(X[:,1].min(), X[:,1].max(), 100).T
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(X1.shape)
    for i in xrange(X1.shape[1]):
       this_X = np.column_stack((X1[:, i], X2[:, i]))
       vals[:, i] = model.predict(gkgm.gaussianKernelGramMatrix(this_X, X))

    plt.contour(X1, X2, vals, colors="blue", levels=[0,0], linewidth=5)
    
    plotData(X, y)