# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 23:10:39 2020

@author: Márton
"""

import numpy as np  # importing numerical package
import scipy as sp  # importing scientific package
import matplotlib.pyplot as plt  # importing MATLAB-like plotting framework
import matplotlib.colors as clr  # importing coloring tools from MatPlotLib
from sklearn import linear_model, neural_network

n = 1000
b0 = 3
b1 = 2
sig = 1
x = np.random.normal(0, 1, n)
z = b0 + b1 * x
p = sp.special.expit(z)
y = np.random.binomial(1, p)

# Pontdiagramok az adatokra
plt.figure(1)
plt.title('Pontdiagram a regressziós egyenessel')
plt.xlabel('x input változó')
plt.ylabel('z látens változó')
xmin = min(x) - 0.3
xmax = max(x) + 0.3
zmin = b0 + b1 * xmin
zmax = b0 + b1 * xmax
plt.scatter(x, z, color='blue')
plt.plot([xmin, xmax], [zmin, zmax], color='black')
plt.show()

plt.figure(2)
plt.title('Logisztikus függvény')
plt.xlabel('x')
plt.ylabel('f(x)')
res = 0.0001
base = np.arange(-5, 5, res)
plt.scatter(base, sp.special.expit(base), s=5, color="blue")
plt.show()

plt.figure(3)
plt.title('Pontdiagram az adatokra a látens valószínűséggel')
plt.xlabel('x input')
plt.ylabel('y output')
colors = ['blue', 'red']
plt.scatter(x, p, color="black")
plt.scatter(x, y, c=y, cmap=clr.ListedColormap(colors))
plt.show()

# 2D array input
X = x.reshape(1, -1).T

logreg = linear_model.LogisticRegression()
logreg.fit(X, y)
b0hat1 = logreg.intercept_[0]
b1hat1 = logreg.coef_[0, 0]
score_logreg = logreg.score(X, y)
y_pred_logreg = logreg.predict(X)
p_pred_logreg = logreg.predict_proba(X)

plt.figure(4)
plt.title('Pontdiagram az adatokra a látens valószínűséggel')
plt.xlabel('p input')
plt.ylabel('y output')
plt.scatter(p, p_pred_logreg[:, 1], color='blue')
plt.plot([0, 1], [0, 1], color='black')
plt.show()

perceptron = linear_model.Perceptron()
perceptron.fit(X, y)
b0hat2 = perceptron.intercept_[0]
b1hat2 = perceptron.coef_[0, 0]
score_perceptron = logreg.score(X, y)
y_pred_perceptron = perceptron.predict(X)

neural = neural_network.MLPClassifier(hidden_layer_sizes=(), activation='logistic', solver='lbfgs')
neural.fit(X, y)
b0hat3 = neural.intercepts_[0][0]
b1hat3 = neural.coefs_[0][0, 0]
score_neural = neural.score(X, y)
y_pred_neural = neural.predict(X)
p_pred_neural = neural.predict_proba(X)

plt.figure(5)
plt.title('Pontdiagram az adatokra a látens valószínűséggel')
plt.xlabel('p input')
plt.ylabel('y output')
plt.scatter(p, p_pred_neural[:, 1], color='blue')
plt.plot([0, 1], [0, 1], color='black')
plt.show()

neural1 = neural_network.MLPClassifier(hidden_layer_sizes=(1), activation='logistic', solver='lbfgs')
neural1.fit(X, y)
b0hat41 = neural1.intercepts_[0][0]
b0hat42 = neural1.intercepts_[1][0]
b1hat41 = neural1.coefs_[0][0, 0]
b1hat42 = neural1.coefs_[1][0, 0]
score_neural1 = neural1.score(X, y)
