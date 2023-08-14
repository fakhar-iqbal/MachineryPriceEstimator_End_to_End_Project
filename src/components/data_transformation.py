
### we will not create a preprocessor here, 
###but create a method  of rpeprocessing in utils 
###that will direct;y transform the data into scaled data by using
###  fastai adddatepart and tabular pandas


import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np

class DataTransform:
    def __init__(self):
        pass

    def initiate_data_transform(self,train_data_path,test_data_path):
        try:
            
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("train and test data receieved from data paths")

            X_train = train_df.drop('SalePrice',axis=1)
            y_train = train_df['SalePrice']

            X_test = test_df.drop('SalePrice',axis=1)
            y_test = test_df['SalePrice']

            logging.info("data separated into input and target features")

            train_arr = np.c_[
                np.array(X_train), np.array(y_train)
            ]
            test_arr = np.c_[
                np.array(X_test), np.array(y_test)
            ]

            logging.info("train and test array initiated")

            return (
                train_arr,test_arr
            )

        except Exception as e:
            raise CustomException(e,sys)