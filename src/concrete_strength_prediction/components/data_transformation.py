import os
import sys
import pickle
import numpy as np
import pandas as pd
from concrete_strength_prediction.exception import customexception
from concrete_strength_prediction.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from concrete_strength_prediction.utils.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

        
    
    def get_data_transformation(self):
        
        try:
            logging.info('Data Transformation initiated')

            # Define which columns should be scaled
            numerical_columns = ['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_aggregate', 'fine_aggregate ', 'age']
            
            
            logging.info('Pipeline Initiated')
            
            

            num_pipeline = Pipeline(
            steps=[
            ('imputer', SimpleImputer(strategy='mean')),  
            ('scaler', StandardScaler())])

            # Full Pipeline
            preprocessor = ColumnTransformer(
            transformers=[
            ('num', num_pipeline, numerical_columns)])
            
            return preprocessor
            

        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise customexception(e,sys)

    def remove_outliers_IQR(self,col,df):
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            return df

        except Exception as e:
            logging.info("Problem in Outliers handling code")
            raise CustomException(e,sys)
            
    
    def initialize_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading train and test data from CSV files")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data complete")
            logging.info(f'Train Dataframe Head:\n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head:\n{test_df.head().to_string()}')

            numerical_columns = ['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_aggregate', 'fine_aggregate ', 'age']
            
            logging.info('Outliers removed in train data')
            logging.info(f'train data shape : {train_df.shape}')
            for col in numerical_columns:
                train_df = self.remove_outliers_IQR(col = col, df = train_df)
            logging.info(f'after removed outliers: {train_df.shape}')

            logging.info('Outliers removed in test data')
            logging.info(f'train data shape : {test_df.shape}')
            for col in numerical_columns:
                test_df = self.remove_outliers_IQR(col = col, df = test_df)
            logging.info(f'after removed outliers: {test_df.shape}')

            preprocessing_obj = self.get_data_transformation()

            target_column_name = 'concrete_compressive_strength'
            drop_columns = [target_column_name]

            

            

            
            
            logging.info("Extracting features and target columns")
            # Extract features and target columns
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Read train and test data complete")
            logging.info(f'Train Dataframe Head:\n{input_feature_train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head:\n{input_feature_test_df.head().to_string()}')


            logging.info("Applying preprocessing object on training and testing datasets")
            # Apply preprocessing object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Combining features and target columns into arrays")
            # Combine features and target columns into arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing pickle file")
            # Save preprocessing pickle file
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)

            logging.info("Preprocessing pickle file saved")

            return train_arr, test_arr

        except Exception as e:
            logging.error(f"Exception occurred in the initialize_data_transformation: {e}")
            raise customexception(e, sys)