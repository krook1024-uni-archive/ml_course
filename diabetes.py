# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 11:05:41 2019

@author: Márton
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, model_selection

diabetes = datasets.load_diabetes()
n = diabetes.data.shape[0]
p = diabetes.data.shape[1]

# Particionálás tanító és taszt adatállományra
X_train, X_test, y_train, y_test = model_selection.train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=2019)

# Tanítás scikit-learn osztállyal
reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)
intercept = reg.intercept_
coef = reg.coef_
score_train = reg.score(X_train,y_train)
score_test = reg.score(X_test,y_test)
y_test_pred = reg.predict(X_test)

# A célváltozó (target) igazi és előrejelzett értékének az összehasonlítása
plt.figure(1)
plt.title('Diabetes prediction')
plt.xlabel('True disease progression')
plt.ylabel('Predicted disease progression')
plt.scatter(y_test,y_test_pred,color="blue")
plt.plot([50,350],[50,350],color='red')
plt.show()

# Előrejelzés
pred = reg.predict(diabetes.data)

pred1 = intercept*np.ones((n))+np.dot(diabetes.data,coef)

error = diabetes.target-pred1
centered_target = diabetes.target-diabetes.target.mean()
score1 = 1-np.dot(error,error)/np.dot(centered_target,centered_target)

diabetes2 = datasets.load_diabetes(return_X_y=True)

record = 10
feature = 2
print(diabetes.feature_names[feature],':', diabetes.data[record,feature])
