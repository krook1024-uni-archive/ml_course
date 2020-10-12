# generált adatok

# n        - rekordok száma
# B_0      = 3
# B_1      = 2
# sig^2    = 0.1
# X        ~ normális (0,1)
# eps      ~ normális (0, \sigma^2)
# y        = B_0 + B_1 X x + eps

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection

# random adatok
n = 100
b0 = 3
b1 = 2
b2 = 4
sig = 0.5
X1 = np.random.normal(size=[n, 1])
X2 = np.random.normal(size=[n, 1])
eps = np.random.normal(scale=sig, size=[n, 1])
y = b0 + b1 * X1 + b2 * X2 + eps

# lineáris regresszió
reg = linear_model.LinearRegression()
reg.fit(X, y)
intercept = reg.intercept_ # should match b0
coef = reg.coef_ # should match b1
score = reg.score(X, y)
X_test_pred = reg.predict(X)
y_test_pred = reg.predict(y)

n_test = 100000
y_test = b0 + b1 * xtest

# pontdiagram
plt.figure(1)
plt.title('Pontdiagram regressziós egyenessel')
plt.xlabel('x input')
plt.ylabel('y input')
plt.scatter(X, y, color="blue")
xrange = [np.amin(X) - 0.1, np.amax(X) + 0.1]
yrange = [np.amin(y) - 0.1, np.amax(y) + 0.1]
plt.plot(xrange, yrange, color='red')
plt.plot(y, y_test_pred, color='green')
plt.show()
