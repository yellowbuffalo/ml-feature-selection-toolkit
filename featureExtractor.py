import copy
import numpy as np
import pandas as pd
from datetime import datetime
import sklearn
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class Extractor:
    """
    The class Extractor is to figure out the feature importance for the training data.
    It's a feature selection tool for data modeling.

    x_feature: x features of the training data.
    target: Ground truth(y) of the training data.
    clf: Model to use to figure out the importance.
    feature_importances: The final result of feature importance.
    """
    def __init__(self, x, y): # Setting initial class
        self.x_feature = x
        self.target = y
        self.clf = RandomForestClassifier(n_estimators = 150, class_weight = "balanced_subsample", 
                            criterion = "entropy", n_jobs = -1, random_state=0, 
                             min_samples_leaf = 3, min_samples_split = 2) 
        self.feature_importances = []

    def gridSearch(self, parameters): # Build gridSearch function
        RF = RandomForestClassifier(n_estimators = 150, class_weight = "balanced_subsample", 
                            criterion = "entropy", n_jobs = -1, random_state=0)
        clf = GridSearchCV(RF, parameters, scoring = 'f1_macro', n_jobs = -1)
        print('Start grid search for RF...')
        start_time = datetime.now()
        clf.fit(self.x_feature, self.target)
        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))
        self.clf = clf.best_estimator_ # Get the best estimator

    def training(self, ): # Setting training function
   	    print('Start training...')
   	    start_time = datetime.now()
   	    self.clf.fit(self.x_feature, self.target)
   	    end_time = datetime.now()
   	    print('Duration: {}'.format(end_time - start_time))
   	    self.feature_importances = self.clf.feature_importances_ # Get importance for every feature
