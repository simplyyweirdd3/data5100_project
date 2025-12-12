## Predicting Lifestyle-Related Health Outcomes Using Machine Learning
# DATA 5100 Final Project  
# Ruman Sidhu & Devlin Hoang

## Project Overview
This project uses machine learning models to predict three health outcomes: diabetes, cardiovascular disease, and stroke. We built separate workflows for each dataset and also evaluated a combined dataset to see whether broader population diversity improves prediction.

We compared Logistic Regression and Random Forest models and evaluated them using ROC AUC, accuracy, precision, recall, calibration, and feature importance.

## Repository Structure
README.md

data/
cardio_train.csv
diabetes_prediction_dataset.csv
healthcare-dataset-stroke-data.csv

Notebooks/
Analysis code.ipynb

Outputs/
diabetes outcome.png
heatmap.png
latest_metrics.txt
rf_calibration.png
rf_roc.png

src/
diabetes_random_forest.py

## How to Run the Notebook
1. Keep all datasets inside the data folder.  
2. Open Notebooks/Analysis code.ipynb.  
3. Update the data path if needed:
4. Run all cells in order to reproduce the EDA, modeling, and evaluation steps.
The script in src contains a simple Random Forest workflow for diabetes.

## Outputs
The Outputs folder includes ROC curves, calibration plots, metrics files, and other visual results.

## Datasets Used
Diabetes Prediction Dataset  
https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

Cardiovascular Disease Dataset  
https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

Stroke Prediction Dataset  
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

## Tools and Libraries
Python  
pandas  
numpy  
scikit-learn  
matplotlib  
seaborn  

## License
This project is distributed under the MIT License.  
See the LICENSE file for details.
