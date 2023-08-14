import os
import sys
from src.exception import CustomException
from src.utils import load_obj
from fastai.tabular.all import *
from fastbook import *
import fastbook

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):   #features is passed as df
        try:
            
            model_path = 'artifacts/model.pkl'
            model = load_obj(model_path)

            #preprocessing the data
            features = add_datepart(features,'saledate')
            cats = ['fiModelDesc',
                    'fiBaseModel',
                    'fiSecondaryDesc',
                    'ProductSize',
                    'fiProductClassDesc',
                    'state',
                    'Enclosure',
                    'Transmission',
                    'Ripper',
                    'Tire_Size',
                    'Coupler_System',
                    'Track_Type']
            conts = ['SalesID', 'MachineID', 'ModelID', 'YearMade', 'saleDay', 'saleElapsed']
            procs = [Categorify,FillMissing]

            new_df =TabularPandas(features, procs, cats, conts, y_names=None, splits=None)
            new_df = new_df.train.xs
            scaled_df = new_df[['YearMade', 'fiSecondaryDesc', 'fiProductClassDesc', 'saleElapsed',
       'ModelID', 'Track_Type', 'fiModelDesc', 'Enclosure', 'SalesID',
       'ProductSize', 'Tire_Size', 'MachineID', 'fiBaseModel',
       'Coupler_System', 'state', 'saleDay', 'Ripper', 'Transmission']]

            preds = model.predict(scaled_df)

            return preds



        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 YearMade:int,fiSecondaryDesc, fiProductClassDesc,saleDate,
       ModelID, Track_Type, fiModelDesc, Enclosure, SalesID,
       ProductSize, Tire_Size, MachineID, fiBaseModel,
       Coupler_System, state, Ripper, Transmission):
        

        self.YearMade = YearMade
        self.fiSecondaryDesc = fiSecondaryDesc
        self.fiProductClassDesc=fiProductClassDesc
        self.saleDate=saleDate
        self.ModelID=ModelID
        self.Track_Type=Track_Type
        self.fiModelDesc=fiModelDesc
        self.Enclosure=Enclosure
        self.SalesID=SalesID
        self.ProductSize=ProductSize
        self.Tire_Size=Tire_Size
        self.MachineID=MachineID
        self.fiBaseModel=fiBaseModel
        self.Coupler_System=Coupler_System
        self.state=state
        self.Ripper=Ripper
        self.Transmission=Transmission

    def get_data_As_dataframe(self):
        try:
            
            custom_data_dict = {
                    'YearMade':[self.YearMade], 'fiSecondaryDesc':[self.fiSecondaryDesc],
                    'fiProductClassDesc':[self.fiProductClassDesc],
                    'saledate':[self.saleDate],
                    'ModelID':[self.ModelID],
                    'Track_Type':[self.Track_Type],
                      'fiModelDesc':[self.fiModelDesc],
                        'Enclosure':[self.Enclosure],
                          'SalesID':[self.SalesID],
                      'ProductSize':[self.ProductSize],
                       'Tire_Size':[self.Tire_Size],
                      'MachineID':[self.MachineID],
                     'fiBaseModel':[self.fiBaseModel],
                     'Coupler_System':[self.Coupler_System],
                         'state':[self.state],
                      'Ripper':[self.Ripper],
                    'Transmission':[self.Transmission]
            }

            return pd.DataFrame(custom_data_dict)


        except Exception as e:
            raise CustomException(e,sys)