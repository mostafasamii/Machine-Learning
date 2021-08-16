import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
#import graphviz

dataset = pd.read_csv('iris.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                         algorithm="SAMME.R",
                         n_estimators=100)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

bdt.fit(X_train,y_train)
y_prediction = bdt.predict(X_test)
accuracy=np.mean(y_prediction == y_test)*100
print("The achieved accuracy using Adaboost is " + str(accuracy))
error = []
clf = tree.DecisionTreeClassifier(max_depth=100)
clf.fit(X_train,y_train)
y_prediction = clf.predict(X_test)
accuracy=np.mean(y_prediction == y_test)*100
print("The achieved accuracy using Decision Tree is " + str(accuracy))

