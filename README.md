
# Academic Achievement Metric Estimator

This project predicts student math scores based on various factors such as gender, race/ethnicity, parental education, and other features. The project is built using Python, Flask, and machine learning models. It provides a web interface for users to input student data and receive predicted scores for their math exams.

The project is deployed using **AWS Elastic Beanstalk** with **Continuous Deployment (CD)** pipeline configured using **AWS CodePipeline** with GitHub as the source repository.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Modeling Process](#modeling-process)
- [Deployment](#deployment)
- [Technologies Used](#technologies-used)
- [License](#license)

## Installation

### Prerequisites and Local Run Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/Gouranga-GH/ETEP-ModularCoding-GitHub-CodePipe-AWSEB.git
   cd ETEP-ModularCoding-GitHub-CodePipe-AWSEB
   ```

2. If you're developing locally, create a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

4. To run the Flask app locally, use the following command:

   ```bash
   python application.py
   ```

5. Access the web app at `http://127.0.0.1:5000/` in your browser.

## Usage

### Web Application

1. Run the Flask server locally:

   ```bash
   python application.py
   ```

2. Open the app in your browser at `http://127.0.0.1:5000/`.

3. Enter the required details on the prediction page, such as gender, race, parental education, etc., and click "Predict your Maths Score."

4. The app will display the predicted math score based on the input data.

## Modeling Process

1. **Data Ingestion**: The data is ingested from a CSV file (`stud.csv`) and split into training and test sets using `train_test_split`.

2. **Data Transformation**: Numerical and categorical features are preprocessed using pipelines defined in `data_transformation.py`. Imputation, one-hot encoding, and scaling are applied as needed.

3. **Model Training**: Various models like Random Forest, Decision Tree, and CatBoost are trained using hyperparameter tuning through `GridSearchCV`. The best model is selected based on performance.

4. **Prediction**: The trained model is saved as a pickle file, and predictions are made using new input data through a web interface or directly via the pipeline.

## Deployment

### AWS Elastic Beanstalk

The project is deployed to **AWS Elastic Beanstalk**, which provides an easy-to-use cloud service for deploying and managing applications. The configuration for Elastic Beanstalk is located in the `.ebextensions/` directory.

The application will be accessible via the AWS Elastic Beanstalk URL provided after deployment.

### Continuous Deployment with AWS CodePipeline

This project uses **AWS CodePipeline** to automate the deployment process. The CD pipeline is configured to work with GitHub as the source repository.

- **Source Stage**: Code is pulled from the GitHub repository
- **Deploy Stage**: The application is automatically deployed to AWS Elastic Beanstalk

### Pipeline Configuration

The deployment pipeline follows this flow:
1. **GitHub Repository** → Source code management
2. **AWS CodePipeline** → CD orchestration
3. **AWS Elastic Beanstalk** → Application deployment and hosting

The pipeline automatically triggers on code changes pushed to the main branch, ensuring continuous deployment of updates.

## Technologies Used

- **Flask**: Web framework for the application.
- **Pandas, NumPy**: Data manipulation and analysis.
- **scikit-learn**: Machine learning model training and evaluation.
- **XGBoost, CatBoost**: Advanced gradient boosting models.
- **Jupyter Notebooks**: For EDA and model training.
- **AWS Elastic Beanstalk**: Cloud platform for deploying the application.
- **AWS CodePipeline**: CD automation tool for deployment.
- **GitHub**: Source code repository and version control.

## App Images

![img 1](screenshots/Deployed_Homepage.png)

![img 2](screenshots/Deployed_Default.png)

![img 3](screenshots/Deployed_AWS_Connection.png)

![img 4](screenshots/Deployed_Predicted_Value.png)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
