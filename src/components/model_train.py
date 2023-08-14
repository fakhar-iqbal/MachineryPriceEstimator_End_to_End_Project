import sys
import os
from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass
from src.utils import evaluate_model,save_obj

### we will not train and test and other algos than random forest. it will be a heavy task. 
### we have already tested the best model that is Random forest. so we will find the best params

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor


@dataclass
class ModelTrainConfig:
    model_train_file_path = os.path.join("artifacts",'model.pkl')

class ModelTrain:
    def __init__(self):
        self.model_train_config = ModelTrainConfig()

    def initiate_model_trainer(self,train_arr, test_arr):
        try:
            
            logging.info("training and testting data split into target and test features")
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            logging.info("hyperparameter tnuing")
            model = RandomForestRegressor(n_jobs=-1)

            params = {
                'n_estimators' : [8,32,64,256],
                #'max_depth': [None, 10, 20, 30],
                #'max_leaf_nodes': [None, 10, 20, 30]
            }

            model_score,best_params,best_model = evaluate_model(X_train,y_train,X_test,y_test,model,params)

            if model_score < 0.6:
                raise CustomException("best model not found")
            logging.info("best model and best params found")

            save_obj(self.model_train_config.model_train_file_path,
                     obj=best_model)
            
            logging.info("best model saved to pkl")

            predicted = best_model.predict(X_test)
            r2_sq = r2_score(y_test,predicted)

            return r2_sq




        except Exception as e:
            raise CustomException(e,sys)
        
