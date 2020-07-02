import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
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

            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state = i)
            partial_result = []
            
            for traning_index, test_index in kfold.split(self.predictors, np.zeros(shape=(self.classe.shape[0], 1))):
                classifier = GaussianNB()
                classifier.fit(self.predictors[traning_index], self.classe[traning_index])
                predictions = classifier.predict(self.predictors[test_index])
                accuracy = accuracy_score(self.classe[test_index], predictions)
                partial_result.append(accuracy)

            partial_result = np.asarray(partial_result)
            media = partial_result.mean()
            self.finals_results[i][0] = round(media, 6)

            
    def decion_tree_results(self):
        
        for i in range(30):

            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state = i)
            partial_result = []
            
            for traning_index, test_index in kfold.split(self.predictors, np.zeros(shape=(self.classe.shape[0], 1))):
                classifier = DecisionTreeClassifier()
                classifier.fit(self.predictors[traning_index], self.classe[traning_index])
                predictions = classifier.predict(self.predictors[test_index])
                accuracy = accuracy_score(self.classe[test_index], predictions)
                partial_result.append(accuracy)

            partial_result = np.asarray(partial_result)
            media = partial_result.mean()
            self.finals_results[i][1] = round(media, 6)
    
    def logistic_regression_results(self):
        
        for i in range(30):

            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state = i)
            partial_result = []
            
            for traning_index, test_index in kfold.split(self.predictors, np.zeros(shape=(self.classe.shape[0], 1))):
                classifier = LogisticRegression()
                classifier.fit(self.predictors[traning_index], self.classe[traning_index])
                predictions = classifier.predict(self.predictors[test_index])
                accuracy = accuracy_score(self.classe[test_index], predictions)
                partial_result.append(accuracy)

            partial_result = np.asarray(partial_result)
            media = partial_result.mean()
            self.finals_results[i][2] = round(media, 6)
    
    def svc_results(self):
        
        for i in range(30):

            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state = i)
            partial_result = []

            for traning_index, test_index in kfold.split(self.predictors, np.zeros(shape=(self.classe.shape[0], 1))):
                classifier = SVC(kernel='rbf', C=2.0, random_state=1)
                classifier.fit(self.predictors[traning_index], self.classe[traning_index])
                predictions = classifier.predict(self.predictors[test_index])
                accuracy = accuracy_score(self.classe[test_index], predictions)
                partial_result.append(accuracy)

            partial_result = np.asarray(partial_result)
            media = partial_result.mean()
            self.finals_results[i][3] = round(media, 6)
    
    def k_neighbors_results(self):
        
        for i in range(30):

            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state = i)
            partial_result = []

            for traning_index, test_index in kfold.split(self.predictors, np.zeros(shape=(self.classe.shape[0], 1))):
                classifier = KNeighborsClassifier()
                classifier.fit(self.predictors[traning_index], self.classe[traning_index])
                predictions = classifier.predict(self.predictors[test_index])
                accuracy = accuracy_score(self.classe[test_index], predictions)
                partial_result.append(accuracy)

            partial_result = np.asarray(partial_result)
            media = partial_result.mean()
            self.finals_results[i][4] = round(media, 6)
   
    
    def random_forestes_results(self):
        
        for i in range(30):

            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state = i)
            partial_result = []

            for traning_index, test_index in kfold.split(self.predictors, np.zeros(shape=(self.classe.shape[0], 1))):
                classifier = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=0)
                classifier.fit(self.predictors[traning_index], self.classe[traning_index])
                predictions = classifier.predict(self.predictors[test_index])
                accuracy = accuracy_score(self.classe[test_index], predictions)
                partial_result.append(accuracy)

            partial_result = np.asarray(partial_result)
            media = partial_result.mean()
            self.finals_results[i][5] = round(media, 6)
        
    def mlp_results(self):
        
        for i in range(30):

            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state = i)
            partial_result = []
            for traning_index, test_index in kfold.split(self.predictors, np.zeros(shape=(self.classe.shape[0], 1))):
                classifier = MLPClassifier(max_iter = 500,
                              tol = 0.000010,
                              batch_size=200, learning_rate_init=0.0001,
                              learning_rate='adaptive')
                classifier.fit(self.predictors[traning_index], self.classe[traning_index])
                predictions = classifier.predict(self.predictors[test_index])
                accuracy = accuracy_score(self.classe[test_index], predictions)
                partial_result.append(accuracy)

            partial_result = np.asarray(partial_result)
            media = partial_result.mean()
            self.finals_results[i][6] = round(media, 6)
        
    def writer_results(self):

        try:
            file = open('cross_val_Kfold.csv', 'w')
            writer = csv.writer(file)
            writer.writerow(('Naive Bayes', 'Decision Tree', 'Logistic regression',
                            'SVM', 'Random forest', 'Neural Networks'))

            for i in range(len(self.finals_results)):
                    writer.writerow(self.finals_results[i])

        except OSError:
            print("Can't open data_results.csv")
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

