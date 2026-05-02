# Inflation Prediction Using Machine Learning

## Author
Duyen Jemima Bachu

## Project Overview
This project applies machine learning techniques to predict inflation rates using a global post-COVID dataset. The goal is to analyze economic indicators and build predictive models that estimate inflation.

## Dataset
The dataset used is:
- global_inflation_post_covid.csv

It contains economic indicators such as GDP, unemployment, and other features related to inflation.

## Machine Learning Models Used
- Linear Regression
- Random Forest Regressor

## Project Structure
inflation-prediction-ml/
│
├── data/        # dataset
├── src/         # model code
├── notebooks/   # data analysis
├── requirements.txt
└── README.md

## How to Run

Step 1: Install dependencies  
pip install -r requirements.txt  

Step 2: Run the model  
cd src  
python model.py  

## Results
The models are evaluated using:
- Mean Squared Error (MSE)
- R² Score

## Notes
Make sure the dataset is placed inside the data/ folder before running the code.
