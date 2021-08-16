import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
#Loading data
data = pd.read_csv('USA_Students.csv')
print(data.describe())
X=data['SAT']
Y=data['GPA']
print(X.shape)
#Plotting
#plt.scatter(X, Y)
#plt.xlabel('SAT', fontsize = 20)
#plt.ylabel('GPA', fontsize = 20)
#plt.show()

cls = linear_model.LinearRegression()
X=np.expand_dims(X, axis=1)
Y=np.expand_dims(Y, axis=1)
cls.fit(X,Y) #Fit method is used for fitting your training data into the model
prediction= cls.predict(X)
plt.scatter(X, Y)
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.plot(X, prediction, color='red', linewidth = 3)
plt.show()
print('Co-efficient of linear regression',cls.coef_)
print('Intercept of linear regression model',cls.intercept_)
print('Mean Square Error', metrics.mean_squared_error(Y, prediction))

#Predict your GPA based on your SAT Score
STA_Score=int(input('Enter your SAT score: '))
x_test=np.array([STA_Score])
x_test=np.expand_dims(x_test, axis=1)
y_test=cls.predict(x_test)
print('Your predicted GPA is ' + str(float(y_test[0])))