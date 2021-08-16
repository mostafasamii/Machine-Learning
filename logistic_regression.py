import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from helper_functions import *

data = pd.read_csv('marks.csv')
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

# filter out the applicants that got admitted
admitted = data.loc[Y == 1]
# filter out the applicants that din't get admission
not_admitted = data.loc[Y == 0]

# plots
plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
plt.show()

#prepare data to be fitted
X = np.c_[np.ones((X.shape[0], 1)), X]
Y = Y[:, np.newaxis]
theta = fit(X, Y)

#plot decision boundary
x_values = [np.min(X[:, 1] - 5), np.max(X[:, 2] + 5)]
theta = theta.flatten()
y_values = - (theta[0] + np.dot(theta[1], x_values)) / theta[2]
plt.plot(x_values, y_values, label='Decision Boundary')
plt.xlabel('Marks in 1st Exam')
plt.ylabel('Marks in 2nd Exam')
plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
plt.show()

actual_classes = Y.flatten()
predicted_classes = predict_classes(theta,X)
predicted_classes = predicted_classes.flatten()
accuracy = np.mean(predicted_classes == actual_classes)
print(accuracy * 100)
