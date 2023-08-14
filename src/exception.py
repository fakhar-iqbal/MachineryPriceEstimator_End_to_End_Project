import sys
import logging
from src.logger import logging


#creating method to design the style of custom exception

def error_message_details(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script [{0}] line number : [{1}] error message: [{2}]".format(
        filename,exc_tb.tb_lineno,str(error)
    )

    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message


if __name__=='__main__':
    logging.info("logger activated in exceptions")