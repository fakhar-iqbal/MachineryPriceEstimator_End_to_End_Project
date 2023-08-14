import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import fastbook

from fastai.tabular.all import *
from fastbook import *
import fastbook
from dataclasses import dataclass
from src.components.data_transformation import DataTransform
from model_train import ModelTrain

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("entered the data ingestion initiation")
        try:
            
            df = pd.read_csv('notebook/data/sales_data.csv')
            logging.info("read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)

            logging.info("dropping redundant and less important features")

            important_features = ['YearMade', 'fiSecondaryDesc', 'fiProductClassDesc', 'saleElapsed',
       'ModelID', 'Track_Type', 'fiModelDesc', 'Enclosure', 'SalesID',
       'ProductSize', 'Tire_Size', 'MachineID', 'fiBaseModel',
       'Coupler_System', 'state', 'saleDay', 'Ripper', 'Transmission']
            
            df = add_datepart(df, 'saledate')

            sizes = 'Large','Large/Medium', 'Medium', 'Small', 'Mini', 'Compact'

            df['ProductSize'] = df['ProductSize'].astype('category')
            df['ProductSize'] = df['ProductSize'].cat.set_categories(sizes, ordered=True)

            procs = [Categorify,FillMissing]

            condition = (df.saleYear<2007) | (df.saleMonth < 10)
            train_idx = np.where(condition)[0]
            test_idx = np.where(~condition)[0]

            splits = (list(train_idx), list(test_idx))

            dep_var='SalePrice'

            cont,cat = cont_cat_split(df,1,dep_var=dep_var)

            to = TabularPandas(df,procs,cat,cont,y_names=dep_var,splits=splits)

            X_train,y_train = to.train.xs,to.train.y
            X_test, y_test = to.valid.xs,to.valid.y
            logging.info("training and testing split initiated")
            X_train = X_train[important_features]
            X_test = X_test[important_features]

            train_Set = pd.concat([X_train,y_train],axis=1)
            test_set = pd.concat([X_test,y_test],axis=1)
            logging.info("data joined")

            train_Set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)

            logging.info("data saved to files")

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomException(e,sys)

if __name__=='__main__':
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransform()

    train_arr,test_arr = data_transformation.initiate_data_transform(train_data,test_data)

    modeltrain = ModelTrain()
    print('r2 score: ', modeltrain.initiate_model_trainer(train_arr,test_arr))

