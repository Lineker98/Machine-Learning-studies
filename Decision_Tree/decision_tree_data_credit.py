#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#Reading data set
base = pd.read_csv("../data_credit.csv")

#Fill the incosistent values with the mean
base.loc[base.age < 0, 'age'] = 40.92

#Split of data base between predictors and classes
predictors = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

#Fiil the missing values with the mean
imputer = SimpleImputer()
imputer = imputer.fit(predictors[:, 1:4])
predictors[:, 1:4] = imputer.transform(predictors[:, 1:4])

#Escalation process of predictor atributes
scaler = StandardScaler()
predictors = scaler.fit_transform(predictors)

#Split of the data set between training and test data
predictors_training, predictors_test, classe_training, classe_test = train_test_split(predictors, classe, test_size=0.25, random_state=0)

#Now, we'll create our instance with decisiontree class , and than, creat our model tree with data training
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(predictors_training, classe_training)

#After criate the tree model, we can make our predictions
predictions = classifier.predict(predictors_test)

#Here, we check results got through predictions
accuaracy = accuracy_score(classe_test, predictions)
matrix = confusion_matrix(classe_test, predictions)

