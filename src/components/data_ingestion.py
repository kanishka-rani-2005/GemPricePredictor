import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion start.')
        try:
            df=pd.read_csv('notebook\\data\\gemstone.csv')
            logging.info('Data Read succesfully.')

            dir=os.path.dirname(self.ingestion_config.raw_data_path)
            os.makedirs(dir,exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('Raw data Saved.')
            logging.info('Train Test Split Begin.')
            train_df,test_df=train_test_split(df,test_size=0.2,random_state=42)

            train_df.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_df.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Data Ingestion Done')

            logging.info('Returning Both Train Data and Test data.')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys) 
        
    
# Run Data ingestion
if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()

    data_transform=DataTransformation()
    train_arr,test_arr,_=data_transform.initiate_data_transformation(train_data_path,test_data_path)

    model_trainer=ModelTrainer()
    model_trainer.initiate_model_training(train_arr,test_arr)
