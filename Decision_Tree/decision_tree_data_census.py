#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
    
#Reading data set
base = pd.read_csv('../data_census.csv')

#Split the atributes between predictors and classes
predictors = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

#Transformatios of predictor atributes from categoricals for discreet attributes
labelencoder_predictors = LabelEncoder()
predictors[:,1] = labelencoder_predictors.fit_transform(predictors[:,1])
predictors[:,3] = labelencoder_predictors.fit_transform(predictors[:,3])
predictors[:,5] = labelencoder_predictors.fit_transform(predictors[:,5])
predictors[:,6] = labelencoder_predictors.fit_transform(predictors[:,6])
predictors[:,7] = labelencoder_predictors.fit_transform(predictors[:,7])
predictors[:,8] = labelencoder_predictors.fit_transform(predictors[:,8])
predictors[:,9] = labelencoder_predictors.fit_transform(predictors[:,9])
predictors[:,13] = labelencoder_predictors.fit_transform(predictors[:,13])
    
#After the previous step, we need transform the dammy atributes genereted by the codification
#because now we've associated unwanted weights.
transformer = ColumnTransformer(transformers=[('test', OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')
predictors = transformer.fit_transform(predictors).toarray()

#Escalation process of predictor atributes
scaler = StandardScaler()
predictors = scaler.fit_transform(predictors)

#Split the data set between training and test data
predictors_training, predictors_test, classe_training, classe_test = train_test_split(predictors, classe, test_size=0.15, random_state=0)

#Now, we'll create our instance with decisiontree class , and than, creat our tree model with data training
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(predictors_training, classe_training)

#After criate the tree model, we can make our predictions
prediction = classifier.predict(predictors_test)
    
#Here, we check results got through predictions
accuracy = accuracy_score(classe_test, prediction)
matriz = confusion_matrix(classe_test, prediction)
