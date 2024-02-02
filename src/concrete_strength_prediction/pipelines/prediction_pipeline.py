import os
import sys
import pandas as pd
from concrete_strength_prediction.exception import customexception
from concrete_strength_prediction.logger import logging
from concrete_strength_prediction.utils.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")
            
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            
            scaled_data = preprocessor.transform(features)
            
            pred = model.predict(scaled_data)
            
            return pred
            
        except Exception as e:
            raise customexception(e, sys)


class CustomData:
    def __init__(self, cement:float, blast_furnace_slag:float, fly_ash, water:float, superplasticizer:float, coarse_aggregate:float, fine_aggregate:float, age:int):

        self.cement = cement
        self.blast_furnace_slag = blast_furnace_slag
        self.fly_ash = fly_ash
        self.water = water
        self.superplasticizer = superplasticizer
        self.coarse_aggregate = coarse_aggregate
        self.fine_aggregate = fine_aggregate
        self.age = age

    def get_data_as_dataframe(self):
        try:
            concrete_data_input_dict = {
                "cement": [self.cement],
                "blast_furnace_slag": [self.blast_furnace_slag],
                "fly_ash": [self.fly_ash],
                "water": [self.water],
                "superplasticizer": [self.superplasticizer],
                "coarse_aggregate": [self.coarse_aggregate],
                "fine_aggregate ": [self.fine_aggregate],
                "age": [self.age]
            }

            df = pd.DataFrame(concrete_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occurred in prediction pipeline')
            raise customexception(e, sys)