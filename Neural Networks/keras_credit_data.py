#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
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

#Split of the data set into training and test data, being the training size = 75% of total
predictors_training, predictors_test, classe_training, classe_test = train_test_split(predictors, classe, test_size=0.25, random_state=0)

# We'll make our sequential neural networks with two hidden layers, that each hidden layers with two perceptrons, will be acticavated
# by relu function, the nn will have three inputs and only one output, being the output activated by sigmoid function
classifier = Sequential()
classifier.add(Dense(units=2, activation='relu', input_dim=3))
classifier.add(Dense(units=2, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#After criate the neural networks model, we can make our predictions
classifier.fit(predictors_training, classe_training, batch_size=10, epochs=200)
predictions = classifier.predict(predictors_test)
predictions = (predictions > 0.5)

#Here, we check the results got through predictions
accuaracy = accuracy_score(classe_test, predictions)
matrix = confusion_matrix(classe_test, predictions)
