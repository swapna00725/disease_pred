import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')

            
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)


            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        fever : int,
        headache : int,
        nausea : int,
        vomiting : int,
        fatigue : int,
        joint_pain : int,
        skin_rash : int,
        cough : int,
        weight_loss : int,
        yellow_eyes : int
        ):

        self.fever = fever

        self.headache = headache

        self.nausea = nausea

        self.vomiting = vomiting

        self.fatigue = fatigue

        self.joint_pain = joint_pain

        self.skin_rash = skin_rash

        self.cough = cough
        self.weight_loss = weight_loss
        self.yellow_eyes = yellow_eyes

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "fever": [self.fever],
                "headache": [self.headache],
                "nausea": [self.nausea],
                "vomiting": [self.vomiting],
                "fatigue": [self.fatigue],
                "joint_pain": [self.joint_pain],
                "skin_rash": [self.skin_rash],
                "cough": [self.cough],
                "weight_loss": [self.weight_loss],
                "yellow_eyes": [self.yellow_eyes],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)