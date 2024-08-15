# Import necessary modules from Flask, numpy, pandas, and sklearn
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Create a Flask application instance
application = Flask(__name__)

# Alias the application instance as 'app'
app = application

## Define route for the home page
@app.route('/')
def index():
    # Render and return the 'index.html' template for the root URL
    return render_template('index.html') 

# Define route for predicting data
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    # Check if the request method is GET
    if request.method == 'GET':
        # Render and return the 'home.html' template for GET requests
        return render_template('home.html')
    else:
        # For POST requests, extract form data and create an instance of CustomData
        data = CustomData(
            gender=request.form.get('gender'),  # Extract 'gender' from form data
            race_ethnicity=request.form.get('ethnicity'),  # Extract 'ethnicity' from form data
            parental_level_of_education=request.form.get('parental_level_of_education'),  # Extract 'parental_level_of_education' from form data
            lunch=request.form.get('lunch'),  # Extract 'lunch' from form data
            test_preparation_course=request.form.get('test_preparation_course'),  # Extract 'test_preparation_course' from form data
            reading_score=float(request.form.get('writing_score')),  # Convert 'writing_score' form data to float and assign to 'reading_score'
            writing_score=float(request.form.get('reading_score'))  # Convert 'reading_score' form data to float and assign to 'writing_score'
        )
        # Convert the data to a DataFrame for prediction
        pred_df = data.get_data_as_data_frame()
        print(pred_df)  # Print DataFrame for debugging
        print("Before Prediction")  # Debugging message before prediction

        # Create an instance of PredictPipeline and make a prediction
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")  # Debugging message during prediction
        results = predict_pipeline.predict(pred_df)  # Get prediction results
        print("After Prediction")  # Debugging message after prediction

        # Render and return the 'home.html' template with the prediction results
        return render_template('home.html', results=results[0])

# Run the Flask application if this script is executed directly
if __name__ == "__main__":
    app.run(host="0.0.0.0")  # Run the app on all available IP addresses
