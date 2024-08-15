# Import necessary modules
import sys
from src.logger import logging  # Import custom logging setup from src.logger

# Function to generate a detailed error message
def error_message_detail(error, error_detail: sys):
    # Extract the traceback information from the exception
    _, _, exc_tb = error_detail.exc_info()
    
    # Get the filename where the exception occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Create a detailed error message including the filename, line number, and error message
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error))

    return error_message

# Custom exception class for handling errors
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        # Call the base class constructor with the error message
        super().__init__(error_message)
        # Generate a detailed error message and store it in an instance variable
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    def __str__(self):
        # Return the detailed error message when the exception is converted to a string
        return self.error_message
