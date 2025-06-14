import os
import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder ,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_object(self):
        try:
            logging.info('Get data Transformation object start.')
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']

            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('Define numerical and categorical columns.')
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical Columns : {categorical_cols}")
            logging.info(f"Numerical Columns : {numerical_cols}")

            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_cols),
                    ('cat_pipeline',cat_pipeline,categorical_cols)
                ]
            )
            logging.info('ColumnTransformer Done.')

            logging.info('Returning preprocessor.')
            return preprocessor
        
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read train and test data completed.')
            
            logging.info('Obtaining preprocessing object.')

            # transformer = DataTransformation()
            preprocessing_obj = self.get_object()

            target_columns='price'
            logging.info(f"Target variable : {target_columns}")

            x_train=train_df.drop(columns=[target_columns,'id'],axis=1)
            y_train=train_df[target_columns]

            x_test=test_df.drop(columns=[target_columns,'id'],axis=1)
            y_test=test_df[target_columns]

            logging.info('Split dependent and independent vars.')

            logging.info('Applying preprocessing object on training and testing datasets.')

            input_feature_train=preprocessing_obj.fit_transform(x_train)
            input_feature_test=preprocessing_obj.transform(x_test)

            train_arr=np.c_[input_feature_train,np.array(y_train)]
            test_arr=np.c_[input_feature_test,np.array(y_test)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('Preprocessor Pickle file Saved.')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path

            )


            
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)