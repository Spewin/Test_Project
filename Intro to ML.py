import numpy as np
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
# print(iris.data)
# print(iris.target)
digits=datasets.load_digits()

svm_model= svm.SVC(gamma=0.001, C=100)

trainX=digits.data[0:-1]
trainy=digits.target[0:-1]

svm_model.fit(trainX,trainy)
print("SVM predict one",svm_model.predict(digits.data[-1:]))
