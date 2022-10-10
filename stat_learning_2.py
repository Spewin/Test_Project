import numpy as np
from sklearn import datasets, svm

## By hand...
X, y = datasets.load_digits(return_X_y=True)
kfolds = 3
X_folds = np.array_split(X, kfolds)
y_folds = np.array_split(y, kfolds)
scores = list()
svc = svm.SVC(kernel='linear')
for k in range(kfolds):
    X_train = list(X_folds)
    X_test = X_train.pop(k)
    X_train = np.concatenate(X_train)
    y_train = list(y_folds)
    y_test = y_train.pop(k)
    y_train = np.concatenate(y_train)
    scores.append(svc.fit(X_train, y_train).score(X_test, y_test))

## using some built ins

from sklearn.model_selection import KFold, cross_val_score

k_fold = KFold(n_splits=3)
for train_ind, test_ind in k_fold.split(X):
    svc.fit(X[train_ind], y[train_ind])
    train_score = svc.score(X[train_ind], y[train_ind])
    test_score = svc.score(X[test_ind], y[test_ind])
    print(f"Train Score: {train_score:.2f}, Test Score: {test_score:.2f}")

##even better using all built in
cross_val_score(svc, X, y, cv=k_fold)

'''Exercise

On the digits dataset, plot the cross-validation score of a SVC estimator with a linear kernel as a function of parameter C (use a logarithmic grid of points, from 1 to 10).'''

import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn import datasets, svm
import matplotlib.pyplot as plt

X, y = datasets.load_digits(return_X_y=True)
k_fold = KFold(n_splits=5)
svc = svm.SVC(kernel="linear")
C_s = np.logspace(-10, 0, 10)

scores = [cross_val_score(svc.set_params(C=c), X, y, n_jobs=-1) for c in C_s]
scores_std = np.std(scores)
means = np.mean(scores, axis=1)
plt.figure()
plt.plot(np.log10(C_s), means)
plt.plot(np.log10(C_s), means + 2 * scores_std, 'b--')
plt.plot(np.log10(C_s), means - 2 * scores_std, 'b--')
plt.show()

'''Exercise

On the diabetes dataset, find the optimal regularization parameter alpha.

Bonus: How much can you trust the selection of alpha?'''
import numpy as np

from sklearn import datasets
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

X, y = datasets.load_diabetes(return_X_y=True)
X = X[:150]
y = y[:150]

model=Lasso()
parameters={'alpha':np.logspace(-4,1,20)}
gridsearch=GridSearchCV(model,parameters)
gridsearch.fit(X,y)

print(gridsearch.best_params_)


from sklearn import cluster, datasets
X,y = datasets.load_iris(return_X_y=True)
k_means=cluster.KMeans(n_clusters=3)
k_means.fit(X)
print(k_means.labels_[::10])
print(y[::10])