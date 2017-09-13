import numpy as np 

def featureNormalize(X):
    X_norm = X
    mu    = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1])) 
    for i in xrange(X.shape[1]):
    	mu[:,i] = np.mean(X[:,i])
    	sigma[:,i] = np.std(X[:,i])
    	X_norm[:,i] = (X[:,i] - float(mu[:,i]))/float(sigma[:,i])

    return X_norm, mu, sigma

# ============================================================
