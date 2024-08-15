# Import necessary modules
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer  # For applying different transformations to different columns
from sklearn.impute import SimpleImputer  # For handling missing values
from sklearn.pipeline import Pipeline  # For creating a sequence of data processing steps
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # For encoding categorical variables and scaling numerical features

from src.exception import CustomException  # Custom exception handling
from src.logger import logging  # Custom logging setup
import os

from src.utils import save_object  # Utility function for saving objects

# Define a data class for data transformation configuration
@dataclass
class DataTransformationConfig:
    # Define the file path for saving the preprocessor object
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

# Class for handling data transformation
class DataTransformation:
    def __init__(self):
        # Initialize the class with the data transformation configuration
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for creating and returning a preprocessing pipeline
        '''
        try:
            # Define the numerical and categorical columns
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Create a pipeline for numerical data: impute missing values and scale features
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # Impute missing values with median
                    ("scaler", StandardScaler())  # Scale features to zero mean and unit variance
                ]
            )

            # Create a pipeline for categorical data: impute missing values, encode categories, and scale features
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Impute missing values with most frequent value
                    ("one_hot_encoder", OneHotEncoder()),  # Encode categorical variables as one-hot vectors
                    ("scaler", StandardScaler(with_mean=False))  # Scale features without centering
                ]
            )

            # Log information about columns being processed
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine the numerical and categorical pipelines into a single transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            # Raise a custom exception if an error occurs
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function performs data transformation on the training and testing data
        '''
        try:
            # Read the training and testing data into DataFrames
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            # Get the preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Define the target column and numerical columns
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Separate features and target variables for training and testing data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # Apply preprocessing to the training and testing feature data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine the processed features and target variables into arrays
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            # Save the preprocessing object to a file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            # Raise a custom exception if an error occurs
            raise CustomException(e, sys)
