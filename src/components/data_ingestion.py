# Import necessary modules
import os
import sys
from src.exception import CustomException  # Custom exception handling
from src.logger import logging  # Custom logging setup
import pandas as pd

from sklearn.model_selection import train_test_split  # Function for splitting data
from dataclasses import dataclass  # For defining data classes

from src.components.data_transformation import DataTransformation  # Data transformation component
from src.components.data_transformation import DataTransformationConfig  # Configuration for data transformation

from src.components.model_trainer import ModelTrainerConfig  # Configuration for model trainer
from src.components.model_trainer import ModelTrainer  # Model trainer component

# Define a data class for data ingestion configuration
@dataclass
class DataIngestionConfig:
    # Define file paths for train, test, and raw data
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

# Class for handling data ingestion
class DataIngestion:
    def __init__(self):
        # Initialize the class with default configuration
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        # Log the start of the data ingestion process
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset into a DataFrame
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            # Create the directory for saving data files if it does not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data to a CSV file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            # Split the data into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the training and testing sets to CSV files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # Return the paths to the training and testing data
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # Raise a custom exception if an error occurs
            raise CustomException(e, sys)

# Main block for executing the data ingestion and processing pipeline
if __name__ == "__main__":
    # Create an instance of DataIngestion and start the data ingestion process
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Create an instance of DataTransformation and start data transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Create an instance of ModelTrainer and start model training
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
