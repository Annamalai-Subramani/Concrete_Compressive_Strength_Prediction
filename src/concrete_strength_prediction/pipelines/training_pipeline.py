import os
import sys
from concrete_strength_prediction.exception import customexception
from concrete_strength_prediction.logger import logging
import pandas as pd
import numpy as np
from concrete_strength_prediction.components.data_ingestion import DataIngestion
from concrete_strength_prediction.components.data_transformation import DataTransformation
from concrete_strength_prediction.components.model_trainer import ModelTrainer

logging.info('Training Pipeline has started')

try:
    obj=DataIngestion()
    train_data_path,test_data_path=obj.Initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr=data_transformation.initialize_data_transformation(train_data_path,test_data_path)

    model_trainer_obj=ModelTrainer()
    model_trainer_obj.initate_model_training(train_arr,test_arr)

except Exception as e:
    raise customexception(e,sys)