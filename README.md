Employee Salary Prediction
This project aims to predict employee salaries based on various features such as years at company, satisfaction level, and average monthly hours using Machine Learning models including Linear Regression, Support Vector Regression (SVR), and Random Forest Regressor.

Table of Contents
Dataset
Features
Data Preprocessing
Exploratory Data Analysis
Modeling
Results
Usage
Files
Requirements
Author
Dataset

The dataset used in this project (employee dataset.csv) contains 1000 employee records with the following columns:

Column	Description
Employee_ID	Unique ID of the employee
Age	Age of the employee
Gender	Gender of the employee
Department	Department the employee works in
Job_Title	Job designation
Years_at_Company	Number of years with the company
Satisfaction_Level	Employee satisfaction level (0 to 1)
Average_Monthly_Hours	Average monthly working hours
Promotion_Last_5Years	Whether promoted in last 5 years (0/1)
Salary	Employee salary (target variable)
Attrition	Employee attrition (0 = No, 1 = Yes)


Features
For predicting salary, the selected features are:
Years_at_Company
Satisfaction_Level
Average_Monthly_Hours
The target variable is: Salary.
Data Preprocessing
Dropped the Employee_ID column as it is not relevant.
Checked for missing values and duplicates (none found).
Standardized numerical features using StandardScaler from scikit-learn.
Exploratory Data Analysis
Pie chart for Gender distribution.
Line plot for average salary by Job Title.
Grouped salary analysis by Department and Promotion_Last_5Years.
Modeling

Three regression models were trained and evaluated:

Linear Regression
Support Vector Regression (SVR)
Hyperparameter tuning performed using GridSearchCV.
Best parameters: {'C': 100, 'degree': 2, 'epsilon': 0.1, 'kernel': 'rbf'}
Random Forest Regressor
Hyperparameter tuning performed using GridSearchCV.
Best parameters: {'max_depth': 5, 'n_estimators': 6}

Results

Model performance on test set:

Model	Mean Absolute Error	Root Mean Squared Error
Linear Regression	18,213.02	20,611.32
SVR	18,158.54	20,522.60
Random Forest	18,497.71	21,125.02
Usage
Clone this repository:
git clone <repository-url>


Install required packages:
pip install -r requirements.txt
Load the pre-trained model:
import joblib
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")
Prepare your input data and scale it:
X_new_scaled = scaler.transform(X_new)
predicted_salary = model.predict(X_new_scaled)

Files
employee dataset.csv – Dataset used for training and testing
model.pkl – Saved Linear Regression model
scaler.pkl – StandardScaler for feature scaling
salary_prediction.ipynb – Jupyter Notebook containing the full workflow

Requirements
Python 3.8+
pandas
numpy
scikit-learn
matplotlib
joblib
