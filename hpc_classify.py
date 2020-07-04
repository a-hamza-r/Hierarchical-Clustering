# Python 3

# CIS5930 Data Mining project
# Remi Trettin, Ameer Hamza, Kenneth Burnham

# This script uses the '10k.anon.csv' file as input
# then trains, tests, and evaluates several classification
# models.

# IMPORTS

import gc
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore") # explicitly ignore any warnings

df = pd.read_csv('10k.anon.csv') # read input data
df = df.sample(frac=1).reset_index(drop=True) # randomly shuffle the data again
print(df[:10].to_string()) # print first 10 samples
print("Number of classes:", len(df['app_name'].unique()))
print("Number of samples:", len(df.index))

# use labelencoder to encode string class names to numbers
le = LabelEncoder()
le.fit(df['app_name'])
print("Encoded classes:", list(le.classes_)) # print classes such that the index of the string class corresponds to its numbered encoding
y = le.transform(df['app_name'])
df = df.assign(app_name=y) # change the dataframe classes to use the new numeric encoding

# split the data into train/test subsets where the test data is 40%, training data is 60%
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 0:7], y, test_size=0.4)

# visualize the class frequency among train/test sets
unique, counts = np.unique(y_train, return_counts=True) # blue
plt.bar(unique, counts)
unique, counts = np.unique(y_test, return_counts=True) # orange
plt.bar(unique, counts)
plt.title('Class Frequency')
plt.xlabel('Class (encoded)')
plt.ylabel('Frequency')
plt.show()

# K-nearest neighbors classifier training/testing with k=5
classifier = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='kd_tree', leaf_size=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
not_predicted = set(y_test) - set(y_pred)
print("*** KNN ***")
print("Classes not predicted:", str(not_predicted), str(len(not_predicted)))
print("F-score:", str(f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))))
print("Accuracy:", str(accuracy_score(y_test, y_pred)))
del classifier
del y_pred
gc.collect()

# Naive bayes (gaussian) classifier training/testing with default parameters
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
not_predicted = set(y_test) - set(y_pred)
print("*** Naive Bayes (Gaussian) ***")
print("Classes not predicted:", str(not_predicted), str(len(not_predicted)))
print("F-score:", str(f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))))
print("Accuracy:", str(accuracy_score(y_test, y_pred)))
del classifier
del y_pred
gc.collect()

# Multilayer perceptron classifier
# Activation function: f(x) = max(0, x)
# Weight optimization: lbfgs because of better performance
classifier = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='lbfgs', max_iter=200, shuffle=True)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
not_predicted = set(y_test) - set(y_pred)
print("*** Multilayer Perceptron ***")
print("Classes not predicted:", str(not_predicted), str(len(not_predicted)))
print("F-score:", str(f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))))
print("Accuracy:", str(accuracy_score(y_test, y_pred)))
del classifier
del y_pred
gc.collect()

# Random forest classifier with 100 trees and GINI split criterion
# Due to class imbalance, use balanced class weights such as: weight = n_samples / (n_classes * count(y))
classifier = RandomForestClassifier(n_estimators=100, criterion='gini', class_weight='balanced')
classifier.fit(X_train, y_train)

# Export one of the decision trees (e.g., number 5) for visualization
estimator = classifier.estimators_[5]
export_graphviz(estimator, out_file='tree.dot', feature_names=list(df.columns[0:7]), class_names=list(le.classes_),
	rounded=True, proportion=False, precision=2, filled=True)

y_pred = classifier.predict(X_test)
not_predicted = set(y_test) - set(y_pred)
print("*** Random Forest ***")
print("Classes not predicted:", str(not_predicted), str(len(not_predicted)))
print("F-score:", str(f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))))
print("Accuracy:", str(accuracy_score(y_test, y_pred)))
del classifier
gc.collect()

# Run random forest again, with cross validation
classifier = RandomForestClassifier(n_estimators=100, criterion='gini', class_weight='balanced')
scores = cross_val_score(classifier, df.iloc[:, 0:7], y, cv=5, scoring='accuracy') # 5-fold cross validation
print("Cross-validated accuracy:", scores)

# Create scatter plot to visualize accuracy of predicted values
_, ax = plt.subplots()
ax.scatter(x=range(0, len(y_test)), y=y_test, c='blue', label='Actual', alpha=0.3)
ax.scatter(x=range(0, len(y_pred)), y=y_pred, c='red', label='Predicted', alpha=0.3)
plt.title('Actual and Predicted Values')
plt.xlabel('Samples')
plt.ylabel('Classes')
plt.legend()
plt.show()