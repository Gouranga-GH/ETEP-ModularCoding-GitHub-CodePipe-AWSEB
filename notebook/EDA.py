# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Ignore warnings for a cleaner output
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('data/stud.csv')

# Display the first few rows of the dataset
print("Top 5 records:")
print(df.head())

# Show the shape of the dataset (rows, columns)
print("\nShape of the dataset:")
print(df.shape)

# Display dataset information including data types
print("\nDataset information:")
print(df.info())

# Display the number of unique values for each column
print("\nNumber of unique values in each column:")
print(df.nunique())

# Display statistical summary of the dataset
print("\nStatistical summary:")
print(df.describe())

# Display unique categories in each categorical column
print("\nCategories in 'gender':", df['gender'].unique())
print("Categories in 'race_ethnicity':", df['race_ethnicity'].unique())
print("Categories in 'parental_level_of_education':", df['parental_level_of_education'].unique())
print("Categories in 'lunch':", df['lunch'].unique())
print("Categories in 'test_preparation_course':", df['test_preparation_course'].unique())

# Define numerical and categorical columns
numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']

# Print the list of numerical and categorical features
print('\nNumerical features:', numeric_features)
print('Categorical features:', categorical_features)

# Add columns for 'Total Score' and 'Average'
df['total score'] = df['math_score'] + df['reading_score'] + df['writing_score']
df['average'] = df['total score'] / 3

# Display the updated DataFrame
print("\nUpdated DataFrame with total and average scores:")
print(df.head())

# Count students with full marks and less than 20 marks in each subject
reading_full = df[df['reading_score'] == 100]['average'].count()
writing_full = df[df['writing_score'] == 100]['average'].count()
math_full = df[df['math_score'] == 100]['average'].count()
reading_less_20 = df[df['reading_score'] <= 20]['average'].count()
writing_less_20 = df[df['writing_score'] <= 20]['average'].count()
math_less_20 = df[df['math_score'] <= 20]['average'].count()

print(f'\nNumber of students with full marks in Maths: {math_full}')
print(f'Number of students with full marks in Writing: {writing_full}')
print(f'Number of students with full marks in Reading: {reading_full}')
print(f'Number of students with less than 20 marks in Maths: {math_less_20}')
print(f'Number of students with less than 20 marks in Writing: {writing_less_20}')
print(f'Number of students with less than 20 marks in Reading: {reading_less_20}')

# Plot histograms and KDE for average and total scores
fig, axs = plt.subplots(1, 2, figsize=(15, 7))
plt.subplot(121)
sns.histplot(data=df, x='average', bins=30, kde=True, color='grey')
plt.subplot(122)
sns.histplot(data=df, x='average', kde=True, hue='gender')
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(15, 7))
plt.subplot(121)
sns.histplot(data=df, x='total score', bins=30, kde=True, color='grey')
plt.subplot(122)
sns.histplot(data=df, x='total score', kde=True, hue='gender')
plt.show()

# Plot histograms for average scores by lunch type and gender
plt.subplots(1, 3, figsize=(25, 6))
plt.subplot(141)
sns.histplot(data=df, x='average', kde=True, hue='lunch')
plt.subplot(142)
sns.histplot(data=df[df.gender == 'female'], x='average', kde=True, hue='lunch')
plt.subplot(143)
sns.histplot(data=df[df.gender == 'male'], x='average', kde=True, hue='lunch')
plt.show()

# Plot histograms for average scores by race/ethnicity and gender
plt.subplots(1, 3, figsize=(25, 6))
plt.subplot(141)
sns.histplot(data=df, x='average', kde=True, hue='race_ethnicity')
plt.subplot(142)
sns.histplot(data=df[df.gender == 'female'], x='average', kde=True, hue='race_ethnicity')
plt.subplot(143)
sns.histplot(data=df[df.gender == 'male'], x='average', kde=True, hue='race_ethnicity')
plt.show()

# Plot violin plots for scores in Math, Reading, and Writing
plt.figure(figsize=(18, 8))
plt.subplot(1, 4, 1)
plt.title('MATH SCORES')
sns.violinplot(y='math_score', data=df, color='grey', linewidth=3)
plt.subplot(1, 4, 2)
plt.title('READING SCORES')
sns.violinplot(y='reading_score', data=df, color='silver', linewidth=3)
plt.subplot(1, 4, 3)
plt.title('WRITING SCORES')
sns.violinplot(y='writing_score', data=df, color='darkgrey', linewidth=3)
plt.show()

# Plot pie charts for categorical features
plt.rcParams['figure.figsize'] = (30, 12)
plt.subplot(1, 5, 1)
size = df['gender'].value_counts()
labels = 'Female', 'Male'
color = ['yellow', 'grey']
plt.pie(size, colors=color, labels=labels, autopct='.%2f%%')
plt.title('Gender', fontsize=20)
plt.axis('off')

plt.subplot(1, 5, 2)
size = df['race_ethnicity'].value_counts()
labels = 'Group C', 'Group D', 'Group B', 'Group E', 'Group A'
color = ['red', 'green', 'blue', 'cyan', 'orange']
plt.pie(size, colors=color, labels=labels, autopct='.%2f%%')
plt.title('Race/Ethnicity', fontsize=20)
plt.axis('off')

plt.subplot(1, 5, 3)
size = df['lunch'].value_counts()
labels = 'Standard', 'Free'
color = ['grey', 'lightblue']
plt.pie(size, colors=color, labels=labels, autopct='.%2f%%')
plt.title('Lunch', fontsize=20)
plt.axis('off')

plt.subplot(1, 5, 4)
size = df['test_preparation_course'].value_counts()
labels = 'None', 'Completed'
color = ['grey', 'silver']
plt.pie(size, colors=color, labels=labels, autopct='.%2f%%')
plt.title('Test Course', fontsize=20)
plt.axis('off')

plt.subplot(1, 5, 5)
size = df['parental_level_of_education'].value_counts()
labels = 'Some College', "Associate's Degree", 'High School', 'Some High School', "Bachelor's Degree", "Master's Degree"
color = ['pink', 'lime', 'lightblue', 'yellow', 'orange', 'grey']
plt.pie(size, colors=color, labels=labels, autopct='.%2f%%')
plt.title('Parental Education', fontsize=20)
plt.axis('off')

plt.tight_layout()
plt.grid()
plt.show()

# Univariate and Bivariate analysis of Gender
f, ax = plt.subplots(1, 2, figsize=(15, 8))
sns.countplot(x=df['gender'], data=df, palette='bright', ax=ax[0], saturation=0.95)
for container in ax[0].containers:
    ax[0].bar_label(container, color='black', size=10)

plt.pie(x=df['gender'].value_counts(), labels=['Male', 'Female'], explode=[0, 0.1], autopct='%1.1f%%', shadow=True)
plt.show()

# Bivariate analysis of Gender impact on performance
numeric_columns = df.select_dtypes(include='number')
gender_group = df.groupby('gender')[numeric_columns.columns].mean()
print("\nMean scores by gender:")
print(gender_group)

X = ['Total Average', 'Math Average']
female_scores = [gender_group['average'][0], gender_group['math_score'][0]]
male_scores = [gender_group['average'][1], gender_group['math_score'][1]]

X_axis = np.arange(len(X))
plt.bar(X_axis - 0.2, male_scores, 0.4, label='Male')
plt.bar(X_axis + 0.2, female_scores, 0.4, label='Female')
plt.xticks(X_axis, X)
plt.ylabel("Marks")
plt.title("Total average vs Math average marks of both genders", fontweight='bold')
plt.legend()
plt.show()

# Univariate and Bivariate analysis of Race/Ethnicity
f, ax = plt.subplots(1, 2, figsize=(15, 8))
sns.countplot(x=df['race_ethnicity'], data=df, palette='bright', ax=ax[0], saturation=0.95)
for container in ax[0].containers:
    ax[0].bar_label(container, color='black', size=10)

plt.pie(x=df['race_ethnicity'].value_counts(), labels=df['race_ethnicity'].unique(), autopct='%1.1f%%', shadow=True)
plt.show()

# Bivariate analysis of Race/Ethnicity impact on performance
race_ethnicity_group = df.groupby('race_ethnicity')[numeric_columns.columns].mean()
print("\nMean scores by race/ethnicity:")
print(race_ethnicity_group)

# Data analysis by parental education level
f, ax = plt.subplots(1, 2, figsize=(15, 8))
sns.countplot(x=df['parental_level_of_education'], data=df, palette='bright', ax=ax[0], saturation=0.95)
for container in ax[0].containers:
    ax[0].bar_label(container, color='black', size=10)

plt.pie(x=df['parental_level_of_education'].value_counts(), labels=df['parental_level_of_education'].unique(), autopct='%1.1f%%', shadow=True)
plt.show()

# Bivariate analysis of parental education impact on performance
education_group = df.groupby('parental_level_of_education')[numeric_columns.columns].mean()
print("\nMean scores by parental level of education:")
print(education_group)
