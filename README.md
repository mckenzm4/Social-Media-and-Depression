# Predicting depression among students based on Social media use

## Overview
This project uses modular coding to build a machine learning pipeline for predicting the mental health state of students based on social media usage. This target variable has values between 4 and 9, but likely was presented as a scale between 1 and 10 on the survey. A higher value indicates a worse mental state. The data is retrieved, cleaned, analyzed, and transformed before being fed into multiple regression algorithms to predict mental health outcomes. The pipeline is designed to be reusable and easily extendable by separating modules into data processing and modeling. The evaluation metrics used are MAE, MSE, RMSE, and R² to assess each model's accuracy and effectiveness. Notebooks have been created for data exploration, cleaning and processing, model training, and model evaluation.

## Features
- Pandas Dataframes
- Scikit-Learn Column Transformer
- Scikit-Learn Linear Regression
- Scikit-Learn Support Vector Regression
- XGBoost XGBRegressor
- TensorFlow Sequential Neural Network

## Project Layout## Project Structure
- **data/** – Raw and processed data  
- **models/** – Model and preprocessor files  
- **notebooks/** – Data exploration, cleaning, model training, and evaluation  
- **src/** – Functions for data processing, modeling, and more  

## Model Evaluation
### Linear Regression
- MAE: 0.2285
- MSE: 0.0998
- RMSE: 0.3159
- R²: 0.9623

### SVR
- MAE: 0.1944
- MSE: 0.0702
- RMSE: 0.2650
- R²: 0.9734

### XGBoost
- MAE: **0.1234**
- MSE: **0.0599**
- RMSE: **0.2448**
- R²: **0.9773**

### Neural Network
- MAE: 0.2587
- MSE: 0.1393
- RMSE: 0.3733
- R²: 0.9473

## Model Selection
The model with the best value for each evaluation metric has been bolded. The XGBoost model yielded the best results among all four metrics for this analysis. By using this model, we can explain 97.73% of the variability in the mental state of the students, and given the datapoints found on the survey, predict the mental state of the students with an average error of 0.1234 points.

The XGBoost Regressor may more accurately predict our target variable because it can detect non-linear relationships within data, which Linear Regression and SVR cannot. To improve the accuracy of these two models, we may consider additional feature engineering or transforming.

The Neural Network model may be overkill for this use case, as they are best used with larger datasets and excel at solving more complex problems.
