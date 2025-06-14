import numpy as np
import pandas as pd
import os 
import sys

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import  Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train,Y_train,X_test,Y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            logging.info('Defining all models.')
            models = {
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "AdaBoostRegressor": AdaBoostRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
                "GradientBoostRegressor":GradientBoostingRegressor()                
            }

            logging.info('Define parameter for hyperparameter tuning.')

            param_grid = {

                
                'Lasso': {
                    'alpha': [0.01, 0.1, 1, 10, 100],
                    'max_iter': [1000, 5000, 10000]
                },

                'Ridge': {
                    'alpha': [0.01, 0.1, 1, 10, 100],
                    'solver': ['auto', 'svd', 'sparse_cg', 'sag']
                },

                'KNeighborsRegressor': {
                    'n_neighbors': [3, 5],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },

                'RandomForestRegressor': {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                },
                
                'DecisionTreeRegressor': {
                    'criterion': ['squared_error', 'friedman_mse'],
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5],
                },

                

                'XGBRegressor': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.7, 1],
                },

                'AdaBoostRegressor': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1]
                },

                'CatBoostRegressor': {
                    'iterations': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'depth': [4, 6],
                    'l2_leaf_reg': [1, 5],
                },

                'SVR': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'poly', 'linear'],
                    'gamma': ['scale', 'auto']
                },

                'GradientBoostRegressor': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                }
            }
            logging.info('Call Evaluate model function')

            model_report:dict = evaluate_model(X_train,Y_train,X_test,Y_test,models,param_grid)

            logging.info('Evaluation done.')
            print(model_report)
            print('*'*50)
            logging.info(f"Model Report\n:{model_report}")

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]
            if best_model_score < 0.6 :
                logging.info('Best model has r2 Score less than 60%')
                raise CustomException('No Best Model Found')
            
            logging.info(f"Best model is : {best_model_name}")
            logging.info(f"Best score is :{best_model_score}")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info('Model saved succesfully.')

            predicted=best_model.predict(X_test)
            score=r2_score(Y_test,predicted)

            logging.info(f'R2 Score is {score}')
            return score

        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys) from e