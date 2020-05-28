#!/usr/bin/env python3
# -*- coding: utf-8 -*-
   
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
    
#Reading database
base = pd.read_csv('../data_census.csv')

#Split the atributes between predictors and classes
predictors = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values
    
#Transformatios of predictor atributes from categorical for discreet atributes
labelencoder_predictors = LabelEncoder()
predictors[:,1] = labelencoder_predictors.fit_transform(predictors[:,1])
predictors[:,3] = labelencoder_predictors.fit_transform(predictors[:,3])
predictors[:,5] = labelencoder_predictors.fit_transform(predictors[:,5])
predictors[:,6] = labelencoder_predictors.fit_transform(predictors[:,6])
predictors[:,7] = labelencoder_predictors.fit_transform(predictors[:,7])
predictors[:,8] = labelencoder_predictors.fit_transform(predictors[:,8])
predictors[:,9] = labelencoder_predictors.fit_transform(predictors[:,9])
predictors[:,13] = labelencoder_predictors.fit_transform(predictors[:,13])
    
#After the previous step, we need transform the dammy atributes genereted by the codification,
#because now we've associated unwanted weights.
transformer = ColumnTransformer(transformers=[('test', OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')
predictors = transformer.fit_transform(predictors).toarray()
    
#Transformation of class atributes
labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)
    
#Escalation process of predictor atributes
scaler = StandardScaler()
predictors = scaler.fit_transform(predictors)

#Split the database into training and test data
predictors_training, predictors_test, classe_training, classe_test = train_test_split(predictors, classe, test_size=0.15, random_state=0)
    
#Base line classifier
print(Counter(classe_test))
    
#We need criate our probabilistic matrix that will genereted by naive-bayes algorithm
classifier = GaussianNB()
classifier.fit(predictors_training, classe_training)

#After criate the matrix, we can make our predictions
prediction = classifier.predict(predictors_test)
    
#Here, we check results got through predictions
accuracy = accuracy_score(classe_test, prediction)
matriz = confusion_matrix(classe_test, prediction)
    
