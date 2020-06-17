#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#Reading of the data set
base = pd.read_csv('../data_credit.csv')

#Tratament the inconsistent values
print(base['age'][base.age > 0].mean())

#Fill the inconsistent values with the mean
base.loc[base.age < 0, 'age'] = 40.92

#Tratament of the missing values
#Split the atributes between predictors and classes
predictors = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

#Fiil the missing values with the mean
imputer = SimpleImputer()
imputer = imputer.fit(predictors[:, 1:4])
predictors[:, 1:4] = imputer.transform(predictors[:, 1:4])

#Schelduling of predictor atributes
scaler = StandardScaler()
predictors = scaler.fit_transform(predictors)

#Split of the data set into training and test data
predictors_training, predictors_test, classe_training, classe_test = train_test_split(predictors, classe, test_size=0.25, random_state=0)

#Now, we'll creat our neural networks model
classifier = MLPClassifier(max_iter=500, tol=0.00001, learning_rate='adaptive')
classifier.fit(predictors_training, classe_training)

#After criate the neural networks model, we can make our predictions
predictions = classifier.predict(predictors_test)

#Here, we check the results got through predictions
accuaracy = accuracy_score(classe_test, predictions)
matrix = confusion_matrix(classe_test, predictions)
