import numpy as np

## Support Vector Machine
def fit(X,Y):
    X=np.c_[np.ones((X.shape[0],1)),X]
    w = np.zeros((X.shape[1],1))
    epochs = 1
    alpha = 0.001

    while (epochs < 10000):
        lambda_ = 1 / epochs
        for SampleIndex,SampleFeatures in enumerate(X):
            SampleFeatures=SampleFeatures.reshape(1,X.shape[1])
            fx = np.dot(SampleFeatures,w)
            if (Y[SampleIndex] * fx  >= 1):
                w = w - alpha * (2 * lambda_ * w)
            else:
                sample_x=np.reshape(X[SampleIndex,:],(X.shape[1],1))
                w = w + alpha *(sample_x * Y[SampleIndex]) - 2 * lambda_ * w
            SampleIndex += 1
        epochs += 1

    y_pred = w[0] + w[1] * X[:, 1] + w[2] * X[:, 2]
    return y_pred,w