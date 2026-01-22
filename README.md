 Exploratory Data Analysis & Machine Learning on Car Prices
 Project Overview

This project performs end-to-end Exploratory Data Analysis (EDA) and Machine Learning modeling on the Car Features and MSRP dataset to understand how different car attributes influence vehicle pricing.

The analysis covers data cleaning, visualization, correlation analysis, outlier detection, and multiple regression models to predict car prices.

 Objectives

Understand relationships between car features and price

Clean and preprocess real-world automotive data

Identify key price-driving factors

Build and evaluate multiple ML regression models

Compare model performance using standard metrics

 Dataset Information

Source: Kaggle – Car Features and MSRP

Records: ~12,000 car models

Time Range: 1990 – 2017

Market: USA

Target Variable: Price

 Dataset link:
https://www.kaggle.com/CooperUnion/cardataset

 Tech Stack & Libraries

Python

NumPy

Pandas

Matplotlib

Seaborn

Scikit-Learn

 Data Preprocessing Steps

Loaded CSV dataset

Removed irrelevant columns

Renamed columns for readability

Identified and removed duplicate rows

Handled missing values

Detected and removed outliers using IQR method

Standardized numerical features

 Exploratory Data Analysis (EDA)

The following analyses were performed:

Distribution analysis of price and numerical features

Missing value percentage visualization

Box plots for outlier detection

Correlation matrix & heatmap

Scatter plots to study feature vs price relationships

Brand popularity and average pricing analysis

 Key Insights

Horsepower (HP) has strong positive correlation with price

Cylinders positively correlate with HP and price

MPG-City & MPG-Highway are negatively correlated with HP

Chevrolet, Toyota, Ford are the most represented brands

Most cars fall within the low-to-mid price range

 Machine Learning Models Implemented

The dataset was split into 80% training / 20% testing, and the following regression models were evaluated:

Linear Regression

Polynomial Regression (Degree = 4)

Decision Tree Regressor

Support Vector Regressor (SVR)

Random Forest Regressor

 Evaluation Metrics

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score

 Random Forest Regressor achieved the best overall performance.

 Project Structure
.
├── exploratory_data_analysis_of_car_features.py
├── README.md
├── requirements.txt
└── data/
    └── cars_data.csv

 How to Run the Project
1. Clone the repository
git clone <your-repo-url>
cd <repo-name>

2. Install dependencies
pip install -r requirements.txt

3. Run the script
python exploratory_data_analysis_of_car_features.py

 Future Improvements

Convert notebook logic into modular functions

Add hyperparameter tuning

Deploy as a web app (Streamlit / Flask)

Add SHAP feature importance

Perform time-based price trend analysis

 Author

Anirudh Pudicheti
 Email: pudichetianirudh@gmail.com

 GitHub: https://github.com/Anirudhpudi

 If you like this project

Give it a ⭐ on GitHub — it really helps!
