from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
data = iris.data
# print(iris.DESCR)
print(data.shape)

import matplotlib.pyplot as plt

digits = datasets.load_digits()
print(digits.images.shape)

# plt.imshow(digits.images[-1], cmap=plt.cm.gray_r)
# plt.show()  # Needed to show plt in my version of PyCharm

data=digits.images.reshape(digits.images.shape[0],-1)
print(data.shape)

iris_X, iris_y = datasets.load_iris(return_X_y= True)
print(np.unique(iris_y))

np.random.seed(0)
indices=np.random.permutation(len(iris_X))
iris_X_train=iris_X[indices[:-10]]
iris_y_train=iris_y[indices[:-10]]
iris_X_test=iris_X[indices[-10:]]
iris_y_test=iris_y[indices[-10:]]

from sklearn.neighbors import KNeighborsClassifier

knn= KNeighborsClassifier()
knn.fit(iris_X_train,iris_y_train)
predictions=knn.predict(iris_X_test)
print(predictions)
print(iris_y_test)

diabetes_X,diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_X_train=diabetes_X[:-20]
diabetes_y_train=diabetes_y[:-20]
diabetes_X_test=diabetes_X[-20:]
diabetes_y_test=diabetes_y[-20:]

from sklearn import linear_model

regression=linear_model.LinearRegression()
regression.fit(diabetes_X_train,diabetes_y_train)
print(regression.coef_)
test_MSE=np.mean((regression.predict(diabetes_X_test)-diabetes_y_test)**2)
train_MSE=np.mean((regression.predict(diabetes_X_train)-diabetes_y_train)**2)
print("Train MSE", train_MSE)
print("Test MSE", test_MSE)
print("R^2 Test", regression.score(diabetes_X_test,diabetes_y_test))



X = np.c_[ .5, 1].T
y = [.5, 1]
test = np.c_[ 0, 2].T
regr = linear_model.LinearRegression()
ridgeregr=linear_model.Ridge(alpha=.1)

import matplotlib.pyplot as plt
# plt.figure()
# np.random.seed(0)
# for _ in range(6):
#     this_X = .1 * np.random.normal(size=(2, 1)) + X
#     regr.fit(this_X, y)
#     plt.plot(test, regr.predict(test))
#     plt.scatter(this_X, y, s=3)
# # plt.show()
#
#
# plt.figure()
# np.random.seed(0)
# for _ in range(6):
#     this_X = .1 * np.random.normal(size=(2, 1)) + X
#     ridgeregr.fit(this_X, y)
#     plt.plot(test, ridgeregr.predict(test))
#     plt.scatter(this_X, y, s=3)
# # plt.show()

alphas=np.logspace(-4,-1,6)
scores=[ridgeregr.set_params(alpha=alpha).fit(diabetes_X_train,diabetes_y_train).score(diabetes_X_test,diabetes_y_test) for alpha in alphas]
plt.figure()
plt.plot(alphas,scores)
# plt.show()

lassoregr=linear_model.Lasso()
scores=[lassoregr.set_params(alpha=alpha).fit(diabetes_X_train,diabetes_y_train).score(diabetes_X_test,diabetes_y_test) for alpha in alphas]
plt.figure()
plt.plot(alphas,scores)
# plt.show()

best_alpha=alphas[np.argmax(scores)]
lassoregr.alpha=best_alpha
lassoregr.fit(diabetes_X_train,diabetes_y_train)
print(lassoregr.coef_)

logistic=linear_model.LogisticRegression(C=1e5)
logistic.fit(iris_X_train,iris_y_train)

'''Exercise: Try classifying the digits dataset with nearest neighbors and a linear model. Leave out the last 10% and test prediction performance on these observations.'''
from sklearn import datasets, neighbors, linear_model
from sklearn.multiclass import OneVsRestClassifier
X_digits, y_digits = datasets.load_digits(return_X_y=True)
X_digits = X_digits / X_digits.max()
n=len(X_digits)
rng=np.random.default_rng(0)
trainindicies=rng.choice(range(n),int(.9*n),replace=False)
trainX=X_digits[trainindicies]
trainy=y_digits[trainindicies]
testX=np.delete(X_digits,trainindicies,axis=0)
testy=np.delete(y_digits,trainindicies,axis=0)

linmodel=linear_model.LogisticRegression()
knnmodel=neighbors.KNeighborsClassifier()

linmodel.fit(trainX,trainy)
knnmodel.fit(trainX,trainy)
print("Linear Score", linmodel.score(testX,testy))
print("KNN Score", knnmodel.score(testX,testy))

from sklearn import svm
rng=np.random.default_rng(0)
indices=np.array([],dtype='int')
iris_X_2=iris_X[:,:2]
for i in np.unique(iris_y):
    indices=np.append(indices,rng.choice(np.arange(len(iris_y), dtype='int')[iris_y==i],int(.9*len(iris_y[iris_y==i])),replace=False))
iris_X_train=iris_X_2[indices]
iris_y_train=iris_y[indices]
iris_X_test=np.delete(iris_X_2,indices,axis=0)
iris_y_test=np.delete(iris_y,indices,axis=0)
for kernel in ['linear', 'rbf', 'poly']:
    svclassifier=svm.SVC(kernel=kernel)
    svclassifier.fit(iris_X_train, iris_y_train)
    svclassifier.score(iris_X_test,iris_y_test)
    print(svclassifier.score(iris_X_test,iris_y_test))
