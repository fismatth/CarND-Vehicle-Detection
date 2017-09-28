from sklearn import svm
from sklearn import grid_search
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import numpy as np
import pickle
import glob

from extract_features import extract_features, merge_features
from sklearn.preprocessing.data import StandardScaler

class Classifier:
    def __init__(self, force_refit=False):
        self.clf_fname = 'vehicle_classifier.p'
        self.scaler_name = 'scaler'
        self.svm_name = 'svm'
        if force_refit:
            self.train_classifier()
        else:
            try:
                # try to load classifier from pickle file
                with open(self.clf_fname, 'rb') as clf_file:
                    data = pickle.load(clf_file)
                    self.svm = data[self.svm_name]
                    self.X_scaler = data[self.scaler_name]
            except:
                self.train_classifier()
    
    def normalize_features(self, X):
        # Apply the scaler to X
        scaled_X = self.X_scaler.transform(X)
        return scaled_X
        
    def train_classifier(self):
        # C small: smooth decision surface
        # gamma large: only few data points relevant
        parameters = {'kernel':('linear',), 'C':[1], 'gamma':[0.1]}
        svr = svm.SVC()
        self.svm = grid_search.GridSearchCV(svr, parameters, verbose=True, n_jobs=8)
        
        cars = glob.glob('data/vehicles/*/*.png')
        notcars = glob.glob('data/non-vehicles/*/*.png')
        
        car_features = extract_features(cars)
        notcar_features = extract_features(notcars)
        
        # Create an array stack of feature vectors
        X = merge_features(car_features, notcar_features)
        # Fit a per-column scaler
        self.X_scaler = StandardScaler().fit(X)
        scaled_X = self.normalize_features(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
        
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_val, y_train, y_val = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
        
        print('train SVM ...')
        self.svm.fit(X_train, y_train)
        print('Best parameters:\n{}'.format(self.svm.best_params_))
        print('Best score:\n{}'.format(self.svm.best_score_))
        
        val_accuracy = self.accuracy(X_val, y_val)
        print('Validation accuracy: {}'.format(val_accuracy))
        
        with open(self.clf_fname, 'wb') as clf_file:
            data = {self.svm_name : self.svm, self.scaler_name : self.X_scaler}
            pickle.dump(data, clf_file)
    
    # classify given features
    def __call__(self, features):
        scaled_features = self.normalize_features(features)
        svm_prediction = self.svm.predict(scaled_features)
        return svm_prediction
    
    def accuracy(self, features, labels):
        return accuracy_score(labels, self(features))


if __name__ == '__main__':
    clf = Classifier(force_refit=True)