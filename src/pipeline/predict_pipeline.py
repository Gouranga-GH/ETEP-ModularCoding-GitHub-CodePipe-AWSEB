# Import necessary modules
import sys
import pandas as pd
from src.exception import CustomException  # Custom exception handling
from src.utils import load_object  # Utility function for loading objects
import os

# Class to handle prediction pipeline
class PredictPipeline:
    def __init__(self):
        # Initialization method, currently does nothing
        pass

    def predict(self, features):
        try:
            # Define paths for the model and preprocessor files
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            print("Before Loading")  # Debugging message before loading objects
            
            # Load the model and preprocessor from disk
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            print("After Loading")  # Debugging message after loading objects
            
            # Apply preprocessing to the input features
            data_scaled = preprocessor.transform(features)
            # Predict using the loaded model
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            # Raise a custom exception if an error occurs
            raise CustomException(e, sys)

# Class to handle custom data input
class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        # Initialize instance variables with input data
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            # Create a dictionary with the input data
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Convert the dictionary to a pandas DataFrame
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            # Raise a custom exception if an error occurs
            raise CustomException(e, sys)
