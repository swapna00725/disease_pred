import os
import sys
from dataclasses import dataclass

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models



@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")

            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Kneighbors": KNeighborsClassifier(),
               
            }
            params={
                "Decision Tree": {
                    'criterion':['gini','entropy'],
                    'max_depth': [None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                },
                "Random Forest":{
                    'criterion': ['gini', 'entropy'],
                    'n_estimators': [10, 50, 100],
                    'max_depth': [None, 5, 10],
                },
                "Kneighbors" :{
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['minkowski', 'manhattan']
                }
                
            }

            model_report, tuned_model = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models, param=params)
            best_model_score = max(model_report.values())

            best_model_name = max(model_report, key=model_report.get)
            best_model = tuned_model[best_model_name]

            logging.info(f"Best Model: {best_model_name} | Accuracy: {best_model_score:.4f}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            acc_score = accuracy_score(y_test, predicted)
            return best_model, acc_score
            



            
        except Exception as e:
            raise CustomException(e,sys)
        
        