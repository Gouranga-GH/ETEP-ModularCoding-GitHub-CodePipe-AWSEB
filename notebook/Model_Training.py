# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Importing Scikit-Learn for modeling
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV

# Importing CatBoost and XGBoost
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# Ignore warnings for a cleaner output
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('data/stud.csv')

# Display the first few records
print("Top 5 records:")
print(df.head())

# Preparing features (X) and target (y) variables
X = df.drop(columns=['math_score'], axis=1)
y = df['math_score']

# Display unique categories in categorical features
print("\nCategories in 'gender':", df['gender'].unique())
print("Categories in 'race_ethnicity':", df['race_ethnicity'].unique())
print("Categories in 'parental_level_of_education':", df['parental_level_of_education'].unique())
print("Categories in 'lunch':", df['lunch'].unique())
print("Categories in 'test_preparation_course':", df['test_preparation_course'].unique())

# Define numeric and categorical features
num_features = X.select_dtypes(exclude="object").columns
cat_features = X.select_dtypes(include="object").columns

# Create Column Transformer with scaling and encoding
numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ("OneHotEncoder", oh_transformer, cat_features),
        ("StandardScaler", numeric_transformer, num_features),
    ]
)

# Apply preprocessing
X = preprocessor.fit_transform(X)
print("\nShape of the transformed feature matrix:", X.shape)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining and testing set shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# Define a function to evaluate model performance
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "XGBRegressor": XGBRegressor(),
    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
    "AdaBoost Regressor": AdaBoostRegressor()
}

# Lists to store model names and R2 scores
model_list = []
r2_list = []

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    model_train_mae, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
    model_test_mae, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)
    
    print(f"{name}:")
    model_list.append(name)
    
    print("Model performance for Training set:")
    print(f"- Root Mean Squared Error: {model_train_rmse:.4f}")
    print(f"- Mean Absolute Error: {model_train_mae:.4f}")
    print(f"- R2 Score: {model_train_r2:.4f}")
    print("Model performance for Test set:")
    print(f"- Root Mean Squared Error: {model_test_rmse:.4f}")
    print(f"- Mean Absolute Error: {model_test_mae:.4f}")
    print(f"- R2 Score: {model_test_r2:.4f}")
    
    r2_list.append(model_test_r2)
    print("="*35)
    print("\n")

# Display results in a DataFrame
results_df = pd.DataFrame(list(zip(model_list, r2_list)), columns=['Model Name', 'R2_Score']).sort_values(by=["R2_Score"], ascending=False)
print("Model Evaluation Results:")
print(results_df)

# Train and evaluate Linear Regression model separately
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred = lin_model.predict(X_test)
score = r2_score(y_test, y_pred) * 100
print(f"\nAccuracy of the Linear Regression model is {score:.2f}%")

# Plot actual vs predicted values for Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
sns.regplot(x=y_test, y=y_pred, ci=None, color='red')
plt.title('Actual vs Predicted Values for Linear Regression')
plt.legend()
plt.show()

# Calculate and display differences between actual and predicted values
pred_df = pd.DataFrame({
    'Actual Value': y_test,
    'Predicted Value': y_pred,
    'Difference': y_test - y_pred
})
print("\nDifference between Actual and Predicted Values:")
print(pred_df.head())
