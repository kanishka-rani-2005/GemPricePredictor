import os
import sys 
from src.exception import CustomException
from src.logger import logging

import dill
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV


def save_object(file_path,obj):
    try :
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as f:
            dill.dump(obj,f)
    except Exception as e:
        logging.error(e)
        raise CustomException(e,sys)
    



def evaluate_model(xtrain,ytrain,xtest,ytest,models,params,cv=3,n_jobs=3,verbose=1,refit=True):
    try:
        report={}

        for i in range(len(list(models))):
            model=list(models.values())[i]
            param_grid=params[list(models.keys())[i]]


            if param_grid: 
                gs = GridSearchCV(model, param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose, refit=refit)
                gs.fit(xtrain, ytrain)
                model.set_params(**gs.best_params_)
                logging.info(f"{model} optimized with GridSearchCV.")
            else:
                logging.info(f"{model} has no params. Skipping GridSearchCV.")

            model.fit(xtrain,ytrain)
            # y_train_pred=model.predict(xtrain)
            y_test_pred=model.predict(xtest)

            # train_model_score=r2_score(ytrain,y_train_pred)
            test_model_score=r2_score(ytest,y_test_pred)

            report[list(models.keys())[i]]=test_model_score

        return report
    
    except Exception as e:
            raise CustomException(f"Model {model} failed during training. Reason: {e}", sys)




def load_object(file_path):
    try:
        with open (file_path,'rb') as f:
            return dill.load(f)
    except Exception as e:
        return CustomException(e,sys)
    