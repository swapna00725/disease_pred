import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report, tuned_model = {}, {}

        for model_name, model in models.items():
            para = param[model_name]
            gs = GridSearchCV(model, para, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
            gs.fit(X_train, y_train)

            # Update with best params
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred, y_test_pred = model.predict(X_train), model.predict(X_test)

            # Accuracy
            train_score = accuracy_score(y_train, y_train_pred)
            test_score = accuracy_score(y_test, y_test_pred)

            # Save results
            report[model_name] = test_score
            tuned_model[model_name] = model

            print(f"[{model_name}] Train: {train_score:.4f} | Test: {test_score:.4f}")
            print(f"Best Params: {gs.best_params_}\n")

        return report, tuned_model
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)