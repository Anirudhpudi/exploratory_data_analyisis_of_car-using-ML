# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis of Car Features (Kaggle: Car Features and MSRP)
Converted from a Colab notebook into a runnable Python script.

IMPORTANT:
1) Install dependencies in terminal (NOT inside this .py file):
   python -m pip install -r requirements.txt

2) Ensure cars_data.csv is in the same folder as this script (or update the path).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional (needed for plots below)
import seaborn as sns

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


def main():
    # 1. Importing the dataset
    data = pd.read_csv("cars_data.csv")

    print("\n--- Head (10) ---")
    print(data.head(10))

    # 1.1 Data types
    print("\n--- Info ---")
    print(data.info())

    # 1.2 Summary stats
    print("\n--- Describe ---")
    print(data.describe(include="all"))

    print("\n--- Shape ---")
    print(data.shape)

    # 2. Dropping irrelevant columns
    cols_to_drop = ['Engine Fuel Type', 'Number of Doors', 'Market Category']
    existing = [c for c in cols_to_drop if c in data.columns]
    data.drop(existing, axis=1, inplace=True)

    print("\n--- After dropping irrelevant columns ---")
    print(data.head(5))

    # 3. Renaming columns
    data.columns = [
        'Make', 'Model', 'Year', 'HP', 'Cylinders', 'Transmission',
        'Drive Mode', 'Vehicle Size', 'Vehicle Style',
        'MPG-H', 'MPG-C', 'Popularity', 'Price'
    ]

    print("\n--- After renaming columns ---")
    print(data.head(5))

    # 4. Dropping duplicate entries
    duplicateDFRow = data[data.duplicated()]
    print("\n--- Duplicate rows (showing first 10) ---")
    print(duplicateDFRow.head(10))

    data.drop_duplicates(keep='first', inplace=True)
    print("\n--- Shape after dropping duplicates ---")
    print(data.shape)

    # 4.1 Missing values
    plt.style.use('ggplot')

    allna = (data.isnull().sum() / len(data)) * 100
    allna = allna.drop(allna[allna == 0].index).sort_values()

    if len(allna) > 0:
        plt.figure(figsize=(8, 4))
        allna.plot.barh()
        plt.title('Missing values percentage per column')
        plt.xlabel('Percentage')
        plt.ylabel('Features with missing values')
        plt.show()

    print("\n--- Null counts per column ---")
    print(data.isnull().sum())

    # Dropping rows with nulls (as per your notebook)
    data = data.dropna()
    print("\n--- Shape after dropping NA rows ---")
    print(data.shape)

    print("\n--- Null counts after dropna ---")
    print(data.isnull().sum())

    # 5. Detecting outliers using boxplots
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 4, 1)
    plt.title("Year Distribution Plot")
    sns.boxplot(x=data['Year'])

    plt.subplot(2, 4, 2)
    plt.title("HP Distribution Plot")
    sns.boxplot(x=data['HP'])

    plt.subplot(2, 4, 3)
    plt.title("Cylinders Distribution Plot")
    sns.boxplot(x=data['Cylinders'])

    plt.subplot(2, 4, 4)
    plt.title("MPG-H Distribution Plot")
    sns.boxplot(x=data['MPG-H'])

    plt.subplot(2, 4, 5)
    plt.title("MPG-C Distribution Plot")
    sns.boxplot(x=data['MPG-C'])

    plt.subplot(2, 4, 6)
    plt.title("Popularity Distribution Plot")
    sns.boxplot(x=data['Popularity'])

    plt.subplot(2, 4, 7)
    plt.title("Price Distribution Plot")
    sns.boxplot(x=data['Price'])

    plt.tight_layout()
    plt.show()

    # IQR method for numeric columns
    df = data.select_dtypes(include=['number'])

    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    print("\n--- Q1 ---")
    print(Q1)
    print("\n--- Q3 ---")
    print(Q3)
    print("\n--- IQR ---")
    print(IQR)

    df1 = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    print("\n--- Shape after removing outliers (numeric-only df1) ---")
    print(df1.shape)

    # 6. Car brands most represented
    brands = data['Make'].value_counts()
    print("\n--- Top 10 brands ---")
    print(brands[:10])

    # Average price of top brands
    average = data[['Make', 'Price']].loc[
        (data['Make'] == 'Chevrolet') | (data['Make'] == 'Ford') |
        (data['Make'] == 'Volkswagen') | (data['Make'] == 'Toyota') |
        (data['Make'] == 'Dodge') | (data['Make'] == 'Nissan') |
        (data['Make'] == 'GMC') | (data['Make'] == 'Honda') |
        (data['Make'] == 'Mazda')
    ].groupby('Make').mean(numeric_only=True)

    print("\n--- Average price of selected top brands ---")
    print(average)

    # 7. Correlation matrix (using df1)
    corr_matrix = df1.corr(numeric_only=True)
    print("\n--- Correlation matrix ---")
    print(corr_matrix)

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

    # 8. Scatter plots: features vs price
    plt.figure(figsize=(20, 12))

    plt.subplot(2, 3, 1)
    plt.title("Year vs Price")
    plt.scatter(data["Year"], data["Price"])
    plt.xlabel("Year")
    plt.ylabel("Price")

    plt.subplot(2, 3, 2)
    plt.title("HP vs Price")
    plt.scatter(data["HP"], data["Price"])
    plt.xlabel("HP")
    plt.ylabel("Price")

    plt.subplot(2, 3, 3)
    plt.title("MPG-H vs Price")
    plt.scatter(data["MPG-H"], data["Price"])
    plt.xlabel("MPG-H")
    plt.ylabel("Price")

    plt.subplot(2, 3, 4)
    plt.title("MPG-C vs Price")
    plt.scatter(data["MPG-C"], data["Price"])
    plt.xlabel("MPG-C")
    plt.ylabel("Price")

    plt.subplot(2, 3, 5)
    plt.title("Popularity vs Price")
    plt.scatter(data["Popularity"], data["Price"])
    plt.xlabel("Popularity")
    plt.ylabel("Price")

    plt.tight_layout()
    plt.show()

    # MPG-H vs MPG-C
    plt.figure(figsize=(8, 4))
    plt.title("MPG-H vs MPG-C")
    plt.scatter(data["MPG-H"], data["MPG-C"])
    plt.xlabel("MPG-H")
    plt.ylabel("MPG-C")
    plt.show()

    # Cylinders vs HP
    plt.figure(figsize=(15, 5))
    plt.scatter(x=data['Cylinders'], y=data['HP'], alpha=0.7)
    plt.title('Cylinders vs HP', weight='bold', fontsize=18)
    plt.xlabel('Cylinders', weight='bold', fontsize=14)
    plt.ylabel('HP', weight='bold', fontsize=14)
    plt.show()

    # HP vs Price regression
    plt.figure(figsize=(15, 5))
    sns.regplot(x=data['HP'], y=data['Price'])
    plt.title('HP vs Price', weight='bold', fontsize=18)
    plt.xlabel('HP', weight='bold', fontsize=14)
    plt.ylabel('Price', weight='bold', fontsize=14)
    plt.show()

    # Cylinders vs Price regression
    plt.figure(figsize=(15, 5))
    sns.regplot(x=data['Cylinders'], y=data['Price'])
    plt.title('Cylinders vs Price', weight='bold', fontsize=18)
    plt.xlabel('Cylinders', weight='bold', fontsize=14)
    plt.ylabel('Price', weight='bold', fontsize=14)
    plt.show()

    # MPG-H/MPG-C vs HP regression
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    sns.regplot(x=data["HP"], y=data["MPG-H"])
    plt.title('MPG-H vs HP', weight='bold', fontsize=18)
    plt.xlabel('HP', weight='bold', fontsize=14)
    plt.ylabel('MPG-H', weight='bold', fontsize=14)

    plt.subplot(1, 2, 2)
    sns.regplot(x=data["HP"], y=data["MPG-C"])
    plt.title('MPG-C vs HP', weight='bold', fontsize=18)
    plt.xlabel('HP', weight='bold', fontsize=14)
    plt.ylabel('MPG-C', weight='bold', fontsize=14)

    plt.tight_layout()
    plt.show()

    print("\n--- Columns ---")
    print(list(data.columns))

    # Countplots
    plt.figure(figsize=(20, 8))

    plt.subplot(1, 2, 1)
    plt.title("Drive Mode")
    sns.countplot(x="Drive Mode", data=data)

    plt.subplot(1, 2, 2)
    plt.title("Vehicle Size")
    sns.countplot(x="Vehicle Size", data=data)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(16, 10))
    plt.title("Vehicle Style by Vehicle Size")
    sns.countplot(y="Vehicle Style", data=data, hue='Vehicle Size')
    plt.xlabel('Count')
    plt.ylabel('Vehicle Style')
    plt.show()

    # Price distribution
    plt.figure(figsize=(16, 5))
    data["Price"].plot.hist(bins=80)
    plt.title("Distribution of Price")
    plt.xlabel('Price Range')
    plt.ylabel('Count')
    plt.show()

    print("\n--- Final Data Head ---")
    print(data.head(3))

    # 9. ML: Split into features and target
    X = data[['Year', 'HP', 'Cylinders', 'MPG-H', 'MPG-C', 'Popularity']].values
    y = data['Price'].values.reshape(-1, 1)

    # Feature Scaling
    scaleX = StandardScaler()
    scaleY = StandardScaler()

    X = scaleX.fit_transform(X)
    y_scaled = scaleY.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_scaled, test_size=0.2, random_state=0
    )

    # 10.1 Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)

    plt.title("Predictions of Linear Regression model")
    plt.xlabel("Y test values")
    plt.ylabel("Y predictions")
    plt.scatter(y_test, y_pred)
    plt.show()

    print("\n--- Linear Regression Metrics ---")
    print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('R2:', metrics.r2_score(y_test, y_pred))

    # 10.2 Polynomial Regression (degree 4)
    poly = PolynomialFeatures(degree=4)
    XP_train = poly.fit_transform(X_train)
    XP_test = poly.transform(X_test)

    poly_reg = LinearRegression()
    poly_reg.fit(XP_train, y_train)
    y_pred_poly = poly_reg.predict(XP_test)

    plt.title("Predictions of Polynomial Regression model")
    plt.xlabel("Y test values")
    plt.ylabel("Y predictions")
    plt.scatter(y_test, y_pred_poly)
    plt.show()

    print("\n--- Polynomial Regression Metrics ---")
    print('MAE:', metrics.mean_absolute_error(y_test, y_pred_poly))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_poly)))
    print('R2:', metrics.r2_score(y_test, y_pred_poly))

    # 10.3 Decision Tree Regressor
    dt_model = DecisionTreeRegressor(random_state=0)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)

    plt.title("Predictions of Decision Tree Regression model")
    plt.xlabel("Y test values")
    plt.ylabel("Y predictions")
    plt.scatter(y_test, y_pred_dt)
    plt.show()

    print("\n--- Decision Tree Metrics ---")
    print('MAE:', metrics.mean_absolute_error(y_test, y_pred_dt))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_dt)))
    print('R2:', metrics.r2_score(y_test, y_pred_dt))

    # 10.4 Support Vector Regressor
    svr_model = SVR(kernel='rbf')
    svr_model.fit(X_train, y_train.ravel())
    y_pred_svr = svr_model.predict(X_test)

    plt.title("Predictions of SVR Regression model")
    plt.xlabel("Y test values")
    plt.ylabel("Y predictions")
    plt.scatter(y_test, y_pred_svr)
    plt.show()

    print("\n--- SVR Metrics ---")
    print('MAE:', metrics.mean_absolute_error(y_test, y_pred_svr))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_svr)))
    print('R2:', metrics.r2_score(y_test, y_pred_svr))

    # 10.5 Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=200, random_state=0)
    rf_model.fit(X_train, y_train.ravel())
    y_pred_rf = rf_model.predict(X_test)

    plt.title("Predictions of Random Forest Regression model")
    plt.xlabel("Y test values")
    plt.ylabel("Y predictions")
    plt.scatter(y_test, y_pred_rf)
    plt.show()

    print("\n--- Random Forest Metrics ---")
    print('MAE:', metrics.mean_absolute_error(y_test, y_pred_rf))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf)))
    print('R2:', metrics.r2_score(y_test, y_pred_rf))


if __name__ == "__main__":
    main()
