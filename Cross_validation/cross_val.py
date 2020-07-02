#!/usr/bin/env python3

import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


class Cross_validation(object):

    def __init__(self):
        self.base = pd.read_csv('../data_credit.csv')
        self.finals_results = [[0 for x in range(7)] for x in range(30)]
    
    def preprocessing(self):

        self.base.loc[self.base.age < 0, 'age'] = self.base['age'][self.base.age > 0].mean()
                
        self.predictors = self.base.iloc[:, 1:4].values
        self.classe = self.base.iloc[:, 4].values

        imputer = SimpleImputer()
        imputer = imputer.fit(self.predictors[:, 1:4])
        self.predictors[:, 1:4] = imputer.transform(self.predictors[:, 1:4])

        scaler = StandardScaler()
        self.predictors = scaler.fit_transform(self.predictors) 
    
    def naive_bayes_results(self):

        for i in range(30):

            partial_result = []
            classifier = GaussianNB()
            cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=i)
            results = cross_val_score(classifier, self.predictors, self.classe, cv=cv)

            partial_result.append(results)
            partial_result = np.asarray(partial_result)
            media = partial_result.mean()
            self.finals_results[i][0] = round(media, 5)

            
    def decion_tree_results(self):
        
        for i in range(30):

            partial_result = []
            classifier = DecisionTreeClassifier()
            cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=i)
            results = cross_val_score(classifier, self.predictors, self.classe, cv=cv)

            partial_result.append(results)
            partial_result = np.asarray(partial_result)
            media = partial_result.mean()
            self.finals_results[i][1] = round(media, 5)

    
    def logistic_regression_results(self):
        
        for i in range(30):

            partial_result = []
            classifier = LogisticRegression()
            cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=i)
            results = cross_val_score(classifier, self.predictors, self.classe, cv=cv)

            partial_result.append(results)
            partial_result = np.asarray(partial_result)
            media = partial_result.mean()
            self.finals_results[i][2] = round(media, 5)

    
    def svc_results(self):
        
        for i in range(30):

            partial_result = []
            classifier = SVC(kernel='rbf', C=2.0, random_state=1)
            cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=i)
            results = cross_val_score(classifier, self.predictors, self.classe, cv=cv)

            partial_result.append(results)
            partial_result = np.asarray(partial_result)
            media = partial_result.mean()
            self.finals_results[i][3] = round(media, 5)
    
    def k_neighbors_results(self):
        
        for i in range(30):

            partial_result = []
            classifier = KNeighborsClassifier()
            cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=i)
            results = cross_val_score(classifier, self.predictors, self.classe, cv=cv)

            partial_result.append(results)
            partial_result = np.asarray(partial_result)
            media = partial_result.mean()
            self.finals_results[i][4] = round(media, 5)
    
    def random_forestes_results(self):
        
        for i in range(30):

            partial_result = []
            classifier = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=0)
            cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=i)
            results = cross_val_score(classifier, self.predictors, self.classe, cv=cv)

            partial_result.append(results)
            partial_result = np.asarray(partial_result)
            media = partial_result.mean()
            self.finals_results[i][5] = round(media, 5)
        
    def mlp_results(self):
        
        for i in range(30):

            partial_result = []
            classifier = MLPClassifier(max_iter = 500,
                              tol = 0.000010,
                              batch_size=200, learning_rate_init=0.0001,
                              learning_rate='adaptive')

            cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=i)
            results = cross_val_score(classifier, self.predictors, self.classe, cv=cv)

            partial_result.append(results)
            partial_result = np.asarray(partial_result)
            media = partial_result.mean()
            self.finals_results[i][6] = round(media, 5)
        
        
    def writer_results(self):

        try:
            file = open('cross_val.csv', 'w')
            writer = csv.writer(file)
            writer.writerow(('Naive Bayes', 'Decision Tree', 'Logistic regression',
                            'SVM', 'Random forest', 'Neural Networks'))

            for i in range(len(self.finals_results)):
                    writer.writerow(self.finals_results[i])

        except OSError:
            print("Can't open cross_val.csv")
        except IndexError:
            print('Index out of range')
        finally:
            file.close()


if __name__ == '__main__':

    validacao = Cross_validation()
    validacao.preprocessing()
    validacao.naive_bayes_results()
    validacao.decion_tree_results()
    validacao.logistic_regression_results()
    validacao.svc_results()
    validacao.k_neighbors_results()
    validacao.random_forestes_results()
    validacao.mlp_results()
    validacao.writer_results()

