# Import necessary modules
import os
import sys
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException  # Custom exception handling

# Function to save an object to a file
def save_object(file_path, obj):
    try:
        # Get the directory path of the file
        dir_path = os.path.dirname(file_path)

        # Create directories if they don't exist
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in binary write mode and save the object
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        # Raise a custom exception if an error occurs
        raise CustomException(e, sys)

# Function to evaluate models using GridSearchCV
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        # Initialize an empty dictionary to store model scores
        report = {}

        # Iterate through each model and its parameters
        for i in range(len(list(models))):
            model = list(models.values())[i]  # Get the model
            para = param[list(models.keys())[i]]  # Get the parameters for the model

            # Perform grid search with cross-validation
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Set the model to the best parameters found
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)  # Train the model

            # Predict on training and test data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R-squared scores for training and test data
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test score in the report dictionary
            report[list(models.keys())[i]] = test_model_score

        # Return the report with model scores
        return report

    except Exception as e:
        # Raise a custom exception if an error occurs
        raise CustomException(e, sys)

# Function to load an object from a file
def load_object(file_path):
    try:
        # Open the file in binary read mode and load the object
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        # Raise a custom exception if an error occurs
        raise CustomException(e, sys)
