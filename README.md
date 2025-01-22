# Disease Prediction using Machine Learning

This project demonstrates a machine learning-based approach for predicting diseases based on patient data. It uses two machine learning models: Logistic Regression and Random Forest, to classify diseases and evaluates their performance.  

## Table of Contents
- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Visualizations](#visualizations)
- [Model Saving and Deployment](#model-saving-and-deployment)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The goal of this project is to predict the likelihood of certain diseases using patient data. The dataset includes attributes such as age, gender, blood pressure, cholesterol levels, and symptoms like fever and fatigue. Two classification models are trained, evaluated, and compared to determine the best performer.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Google Colab (optional, for cloud-based execution)

## Dataset
The dataset used in this project (`Disease_dataset.csv`) contains features such as:
- Age
- Gender
- Blood Pressure
- Cholesterol Level
- Fever
- Cough
- Fatigue
- Difficulty Breathing
- Disease (target variable)

## Features
- **Logistic Regression Model:** Evaluated using accuracy, classification report, and confusion matrix.
- **Random Forest Model:** Evaluated similarly, with additional feature importance analysis.
- **Feature Importance Visualization:** Insights into the contribution of each feature to the predictions.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/disease-prediction-ml.git
   cd disease-prediction-ml
   ```
2. Install required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the dataset (`Disease_dataset.csv`) in the project directory.

## Usage
1. Open the `disease_prediction.py` file in your preferred Python environment.
2. Run the script to preprocess data, train models, and evaluate their performance.
3. Use the `disease_prediction_model.pkl` file to make predictions on new data:
   ```python
   import joblib
   loaded_model = joblib.load('disease_prediction_model.pkl')
   new_data = [[30, 1, 1, 0, 0, 1, 0, 0]]  # Example input
   prediction = loaded_model.predict(new_data)
   print("Predicted Disease:", prediction)
   ```

## Results
- Logistic Regression Accuracy: **X%**  
- Random Forest Accuracy: **Y%**  
(The exact values will depend on the dataset and preprocessing steps.)

## Visualizations
A bar plot visualizing the feature importance for the Random Forest model is generated:
![Feature Importance](path/to/feature-importance-plot.png)

## Model Saving and Deployment
The Random Forest model, identified as the best performer, is saved as `disease_prediction_model.pkl`. This model can be loaded for real-time predictions.
