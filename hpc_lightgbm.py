# Python 3

# CIS5930 Data Mining project
# Remi Trettin, Ameer Hamza, Kenneth Burnham

# This script uses the '10k.anon.csv' file as input
# then trains, tests, and evaluates LightGBM
# ( https://github.com/microsoft/LightGBM )

# IMPORTS

import gc
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore") # explicitly ignore warnings

df = pd.read_csv('10k.anon.csv') # read data
df = df.sample(frac=1).reset_index(drop=True) # shuffle the data again
df['mem_used'] = pd.to_numeric(df['mem_used']) # ensure the data type of the mem_used column is int
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

# Transform the training set into a LightGBM format
d_train = lgb.Dataset(X_train, label=y_train)
d_traincv = lgb.Dataset(df.iloc[:, 0:7], label=y)

# LightGBM model parameters
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'rf' # random forest
params['objective'] = 'multiclass' # 25 classes to choose from
params['metric'] = 'multi_logloss' # use log loss for evaluation
params['feature_fraction'] = 0.8
params['bagging_fraction'] = 0.8
params['bagging_freq'] = 1
params['num_class'] = len(df['app_name'].unique()) # 25
params['verbose'] = -1 # suppress output

# train the classifier over 100 boosting iterations
classifier = lgb.train(params, d_train, 100)

y_pred = classifier.predict(X_test)

# Transform predictions into 1d array
# by using the maximum class probability for each sample
y2pred = []
for i in range(len(y_pred)):
	y2pred.append(list(y_pred[i]).index(max(list(y_pred[i]))))

not_predicted = set(y_test) - set(y2pred)
print("*** LightGBM Random Forest ***")
print("Classes not predicted:", str(not_predicted), str(len(not_predicted)))
print("F-score:", str(f1_score(y_test, y2pred, average='weighted', labels=np.unique(y2pred))))
print("Accuracy:", str(accuracy_score(y_test, y2pred)))

# Test the model again, 5-fold cross validation
scores = lgb.cv(params, d_traincv, nfold=5, stratified=True, shuffle=True, verbose_eval=None, seed=5930)
scores = scores['multi_logloss-mean'] # only interested in the log loss metric
del classifier
gc.collect()

# Graph the log loss metric
plt.plot(np.arange(1, len(scores)+1, 1), scores)
plt.title('LightGBM Multiclass Log Loss (Mean)')
plt.xlabel('Iteration')
plt.ylabel('Mean Log Loss')
plt.show()