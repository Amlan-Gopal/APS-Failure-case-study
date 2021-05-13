import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import re
import pickle
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import joblib

from invalidInputException import InvalidInputException

class ML_Service:

    def __init__(self):
        # Load files
        self.imputer = joblib.load("stored_data/imputer.pkl")
        self.scaler = joblib.load("stored_data/scaler.pkl")
        self.clf = joblib.load("stored_data/lr.sav")
            
    def plot_confusion_matrix(self, test_y, predict_y):
        C = confusion_matrix(test_y, predict_y)
        result = dict()
        misclassified_points = round(((len(test_y)-np.trace(C))/len(test_y)*100),2)
        result['misclassified_points'] = misclassified_points
        print("Number of misclassified points {}%".format(misclassified_points))
        fp = (int)(C[0][1])
        fn = (int)(C[1][0])
        result['false positive'] = fp
        result['false negative'] = fn
        print('Number of False Positives: ', fp)
        print('Number of False Negatives: ', fn)
        cost = (10 * fp) + (500 * fn)
        result['cost'] = cost
        print('Total Cost (cost1+cost2): ', cost)
        log_loss = round(log_loss(y, y_preds), 4)
        result['log loss'] = log_loss
        print('log-loss: ', log_loss)
        f1_score = f1_score(y, y_preds, average='micro')
        result['f1-score'] = f1_score
        print('f1-score: ', f1_score)
        
        
        A =(((C.T)/(C.sum(axis=1))).T)
        
        B =(C/C.sum(axis=0))
        plt.figure(figsize=(20,4))
        
        labels = [0,1]
        # representing A in heatmap format
        cmap=sns.light_palette("blue")
        plt.subplot(1, 3, 1)
        sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted Class')
        plt.ylabel('Original Class')
        plt.title("Confusion matrix")
        
        plt.subplot(1, 3, 2)
        sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted Class')
        plt.ylabel('Original Class')
        plt.title("Precision matrix")
        
        plt.subplot(1, 3, 3)
        # representing B in heatmap format
        sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted Class')
        plt.ylabel('Original Class')
        plt.title("Recall matrix")
        
        plt.show()
        
        return result
        
    def predict(self, X):
        """Give APS prediction for a input list/vector of features"""
        if isinstance(X, list) == False:
            raise InvalidInputException('Please pass valid input. List of values are expected.')
        # Check input length
        if len(X) != 170:
            raise InvalidInputException('Please pass valid input. Some values are missing.')
        X = np.array(X)    
        # Replace non-numeric values with NaN
        X.astype('<U32')
        #Refer: https://stackoverflow.com/questions/16223483/forced-conversion-of-non-numeric-numpy-arrays-with-nan-replacement
        X = np.genfromtxt(X)
        # For a single input, shape should be (1, 170)
        X = X.reshape(1,-1)
        # Replace the missing values using saved imputers if present
        X = self.imputer.transform(X)
        #Standardize the data
        X = self.scaler.transform(X)
        # Remove the column with constant value ('cd_000') index= 89
        X = np.delete(X, 89,1)
        # Predict Y values with logistic regression as classifier
        y_pred = self.clf.predict(X)[0]
        # Label the predicted Y values back into 'pos' and 'neg'
        y_label = 'pos' if y_pred == 1 else 'neg'
        return (y_pred, y_label)

    def checkPerformance(self, X, y):
        """For a given set of X and y values, predict and check model performance"""
        if (isinstance(to_predict_list, list) == False or 
        isinstance(y_list, list) == False):
            raise InvalidInputException('Please pass valid input. List of values are expected for both input')
        #Convert X (list of list) and y(list) to arrays 
        X = np.array(X)
        #Convert Y values ('pos', 'neg' to 1 and 0)
        y = [1 if i == 'pos' else 0 for i in y]
        y = np.array(y)
        size = X.shape[0]
        # Predict for each data points using the selected classifier
        y_preds = []
        y_labels = []
        for i in range(size):
            y_pred, y_label = self.predict(X[i])
            y_preds.append(y_pred)
            y_labels.append(y_label)
        
        # Plot confusion matrix with precision and recall
        result = self.plot_confusion_matrix(y, y_preds)
        result['y_labels'] = y_labels
        return result	