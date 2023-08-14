import os
import sys
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_obj(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj, file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,model,params):
    try:
        
        model=model
        gs = GridSearchCV(model,params,cv=3)
        gs.fit(X_train,y_train)

        model.set_params(**gs.best_params_)
        model.fit(X_train,y_train)

        y_test_pred = model.predict(X_test)
        test_model_score = r2_score(y_test,y_test_pred)

        return test_model_score,model.get_params(),model
        

    except Exception as e:
        raise CustomException(e,sys)


def load_obj(file_path):
    try:
        
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
        

    except Exception as e:
        raise CustomException(e,sys)