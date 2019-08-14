import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

# we load wine quality data
print("\n")
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')
print(data.describe())

# we split the data
# y is the target feature we are trying to predict
y = data.quality
# X is input features so all features execpt quality
X = data.drop('quality', axis=1)
# we split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# we preprocess the data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
print(X_train_scaled.mean(axis=0))
print(X_train_scaled.std(axis=0))

X_test_scaled = scaler.transform(X_test)
print(X_test_scaled.mean(axis=0))
print(X_test_scaled.std(axis=0))

pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

# we declare the hyperparameters
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

# we use a cross-validation for the model
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
# we fit and tune the model
clf.fit(X_train, y_train)

# we evaluate the model pipeline on test data
y_prediction = clf.predict(X_test)
print(r2_score(y_test, y_prediction))
print(mean_squared_error(y_test, y_prediction))

# we save the model to a .pkl file
joblib.dump(clf, 'rf_model.pkl')

# if we want to load the model we use:
# clf2 = joblib.load('rf_model.pkl')
