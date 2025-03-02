# -*- coding: utf-8 -*-
"""Exploratory_Data_Analysis_of_Car_Features.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1f3GxV4DFoFt-Pd_aEf-1Nfdds7ehIUV-

<a href="https://colab.research.google.com/github/DhanaLakshmi2000/EDA_OF_CAR/blob/master/Exploratory_Data_Analysis_of_Car_Features.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# **Project Title:** Exploratory Data Analysis of Car Features!

#**Context**
As a data scientist, the majority of your time will be spent on data pre-processing i.e.
making sure you have the right data in the right format. Once this is done, you get a
sense of your dataset through applying some descriptive statistics and then, you move
on to the exploration stage wherein you plot various graphs and mine the hidden
insights. In this project, you as a data scientist are expected to perform Exploratory data
analysis on how the different features of a car and its price are related. The data comes
from the Kaggle dataset "Car Features and MSRP". It describes almost 12,000 car
models, sold in the USA between 1990 and 2017, with the market price (new or used)
and some features.

#**Objective**
The objective of the project is to do data pre-processing and exploratory data analysis
of the dataset.

**[Click here for the dataset.](https://www.kaggle.com/CooperUnion/cardataset)**

#1.Importing the dataset and the necessary libraries, checking datatype, statistical summary, shape, null values etc.
"""

#importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#files.upload returns a dictionary of the files which were uploaded. The dictionary is keyed by the file name and values are the data which were uploaded.
#uploaded= files.upload()

#Way to load csv file into google colab.
import io

#reading the csv file onto the google colab
data= pd.read_csv('cars_data.csv')

#reading the dataset
data.head(10)

"""##1.1 Checking the datatype of each column"""

#checking the data type of each column of our dataset.
data.info()

"""##1.2 Descriptive or summary statistics in python
 – pandas, can be obtained by using describe function – describe(). Describe Function gives the mean, std and IQR values.**
This function excludes the character columns and gives summary statistics of numeric columns.We need to add a variable named include=’all’ to get the summary statistics or descriptive statistics of both numeric and character column.

"""

#The following function gives the statistical  measures of the numerical values.
data.describe()

"""We can now check the shape of our data set, which means checking the number of rows and columns containing all the information."""

#The result of the following function would be in the form of (number of rows * number of columns)
data.shape

"""# 2.Dropping irrelevant columns
Dropping columns engine fuel type, market category, number of doors as these are irrelevant.
"""

#Dropping irrelevant columns.
data.drop(['Engine Fuel Type', 'Number of Doors','Market Category'], axis=1, inplace=True)
data.head(5)

"""#3.Renaming the columns"""

#Renaming the columns.
data.columns= ['Make', 'Model','Year', 'HP', 'Cylinders','Transmission','Drive Mode', 'Vehicle Size',
       'Vehicle Style', 'MPG-H', 'MPG-C','Popularity', 'Price']
data.head()

"""#4.Dropping Duplicate entries.
In Data Science, we usually come accross some duplicate entries, in the rows. This means two rows having the exact same values for each column. This could be eliminated through the use of pandas.dataframe.duplicated() function.
"""

#Finding the duplicate entries.
duplicateDFRow = data[data.duplicated()]
duplicateDFRow

"""Here all the duplicate rows except their first occurrence are returned because the default value of keep argument was "first".If we want to select all the duplicate rows except their last occurrence, then we need to pass a keep argument as "last" Pandas drop_duplicates() method helps in removing duplicates from the data frame."""

#Removing duplicates from the data frame.
data.drop_duplicates(keep ='first', inplace = True)
#Checking the shape of the dataframe to check whether the duplicates are removed or not.
data.shape

"""##4.1Dropping the missing values
##**Working with Null Values:**
Missing Data can occur when no information is provided for one or more items or for a whole unit. Missing Data is a very big problem in real life scenario. Missing Data can also refer to as NA(Not Available) values.In order to check missing values in Pandas DataFrame, we use a function isnull() and notnull(). Both function help in checking whether a value is NaN or not. These function can also be used in Pandas Series in order to find null values in a series.
"""

!pip install seaborn

import matplotlib.pyplot as plt

plt.style.use('ggplot')

plt.style.use('ggplot')  # Change to an alternative style
allna = (data.isnull().sum() / len(data)) * 100
allna = allna.drop(allna[allna == 0].index).sort_values()
plt.figure(figsize=(8, 4))
allna.plot.barh(color='purple', edgecolor='black')
plt.title('Missing values percentage per column', fontsize=15, weight='bold')
plt.xlabel('Percentage', weight='bold', size=15)
plt.ylabel('Features with missing values', weight='bold')
plt.yticks(weight='bold')
plt.show()

#The following function gives the number of null values, in each column.
data.isnull().sum()

"""After this, we can either drop the rows or columns having null values, or we can fill the missing values with the appropriate values.
We can see the columns, "HP" and "Cylinders", having null values less than 1% of the entire dataset, so we can fill them by ffill method or backword fill method, but we will just drop those values, in order to be more precise and accurate with our results.
"""

#Dropping the missing values.
data=data.dropna()
data.count()

#Checking the number of null values.
data.isnull().sum()

"""**We have successfully removed the null values from our dataset. We can move further with our exploratory data analysis.**

#5.Detecting Outliers.

An outlier is an observation that is unlike the other observations.Outliers are the values in dataset which standouts from the rest of the data. It is rare, or distinct, or does not fit in some way. BOX-PLOTS give out the information on the variability or dispersion of the data.A boxplot is a graph that gives you a good indication of how the values in the data are spread out.
"""

#Importing the seaborn library.
import seaborn as sns
#Plotting the boxplots of each continuous variable to find outliers.
plt.style.use('ggplot')
plt.figure(figsize=(18,12))

plt.subplot(2,4,1)
plt.title("Year Distribution Plot")
sns.boxplot(x=data['Year'], color='teal')

plt.subplot(2,4,2)
plt.title("HP Distribution Plot")
sns.boxplot(x=data['HP'], color='teal')

plt.subplot(2,4,3)
plt.title("Cylinders Distribution Plot")
sns.boxplot(x=data['Cylinders'], color='teal')

plt.subplot(2,4,4)
plt.title("MPG-H Distribution Plot", color='teal')
sns.boxplot(x=data['MPG-H'])

plt.subplot(2,4,5)
plt.title("MPG-C Distribution Plot")
sns.boxplot(x=data['MPG-C'], color='teal')

plt.subplot(2,4,6)
plt.title("Popularity Distribution Plot")
sns.boxplot(x=data['Popularity'], color='teal')

plt.subplot(2,4,7)
plt.title("Price Distribution Plot")
sns.boxplot(x=data['Price'], color='teal')

plt.show()

"""****IQR Method of Outlier Detection****

![iqr.PNG](data:image/png;base64,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)

A box plot tells us, more or less, about the distribution of the data. It gives a sense of how much the data is actually spread about, what’s its range, and about its skewness. As you might have noticed in the figure, that a box plot enables us to draw inference from it for an ordered data, i.e., it tells us about the various metrics of a data arranged in ascending order.
In the above figure,
minimum is the minimum value in the dataset,
and maximum is the maximum value in the dataset.
So the difference between the two tells us about the range of dataset.
The median is the median (or centre point), also called second quartile, of the data (resulting from the fact that the data is ordered).
Q1 is the first quartile of the data, i.e., to say 25% of the data lies between minimum and Q1.
Q3 is the third quartile of the data, i.e., to say 75% of the data lies between minimum and Q3.
The difference between Q3 and Q1 is called the Inter-Quartile Range or IQR.
"""

# Selecting only numeric columns
df = data.select_dtypes(include=['number'])

# Calculating the IQR for numeric columns
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Printing the Q1 and Q3 values
print("Q1:\n", Q1)
print("Q3:\n", Q3)
print("IQR:\n", IQR)

#Removing the outliers from the data.
df1 = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
df1.shape

"""#6.The car brands that are represented the most.

**The  car brands that are the most represented in the dataset.**
"""

#Returning the counts of each Car brand according to their respective number of occurences.
brands=data['Make'].value_counts()
brands[:10]

"""Hence, we can conclude that "Chevrolet" brand is mostly represented in the dataset, followed by "Toyota","Volkswagen","Nissan", "GMC" and so on and so forth"""

#Checking the average price of each of the top ten car brands.
average = data[['Make','Price']].loc[(data['Make'] == 'Chevrolet')|(data['Make'] == 'Ford')|
               (data['Make'] == 'Volkswagen')|(data['Make'] == 'Toyota')|(data['Make'] == 'Dodge')|
               (data['Make'] == 'Nissan')|(data['Make'] == 'GMC')|(data['Make'] == 'Honda')|
               (data['Make'] == 'Mazda')].groupby('Make').mean()
#Printing the average price of top-10 brands.
average

"""#7.Correlation Matrix
Variables within a dataset can be related for lots of reasons.

For example:

One variable could cause or depend on the values of another variable.
One variable could be lightly associated with another variable.
Two variables could depend on a third unknown variable.
It can be useful in data analysis and modeling to better understand the relationships between variables. The statistical relationship between two variables is referred to as their correlation.

A correlation could be positive, meaning both variables move in the same direction, or negative, meaning that when one variable’s value increases, the other variables’ values decrease. Correlation can also be neutral or zero, meaning that the variables are unrelated.

Positive Correlation: both variables change in the same direction.
Neutral Correlation: No relationship in the change of the variables.
Negative Correlation: variables change in opposite directions.
"""

#Creating a correlation matrix.
corr_matrix=df1.corr()
#Printing the matrix.
corr_matrix

"""We can use the seaborn and matplotlib packages in order to get a visual representation of the correlation matrix."""

plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

"""**We can derive the following insights from the correlation matrix which is illustrated below:**


*   HP is highly correlated with Cylinders and Price, and negatively correlated with MPG-H and MPG-C.(This means that Horse Power is the one of the major contributing factors of the price of the car,)
*   Cylinders is in a moderate to high correlation with Price and negatively correlated with MPG-H and MPG-C.(This means the lesser number of cylinders, higher is the MPG-C and MPG-H.)
*   MPG-H and MPG-C are highly correlated among themselves.

#8.Plotting graphs of various columns for better exploratory data analysis.

**Plotting each continuous variable/column of the dataset with then Price column to see what is the correlation between them.**
We will do this task of visualization with the help of Scatter Plot. Scatter plot is a graph in which the values of two variables are plotted along two axes. It is a most basic type of plot that helps you visualize the relationship between two variables.
"""

#Plotting the boxplots of each continuous variable to find if any outlier is left.
plt.figure(figsize=(20,12))

plt.subplot(2,3,1)
plt.title("Year vs Price")
plt.scatter(data["Year"], data["Price"],c=data['Price'], cmap='autumn')
plt.xlabel("Year")
plt.ylabel("Price")

plt.subplot(2,3,2)
plt.title("HP vs Price")
plt.scatter(data["HP"], data["Price"],c=data['Price'], cmap='autumn')
plt.xlabel("HP")
plt.ylabel("Price")

plt.subplot(2,3,3)
plt.title("MPG-H vs Price")
plt.scatter(data["MPG-H"], data["Price"],c=data['Price'], cmap='autumn')
plt.xlabel("MPG-H")
plt.ylabel("Price")

plt.subplot(2,3,4)
plt.title("MPG-C vs Price")
plt.scatter(data["MPG-C"], data["Price"],c=data['Price'], cmap='autumn')
plt.xlabel("MPG-C")
plt.ylabel("Price")

plt.subplot(2,3,5)
plt.title("Popularity vs Price")
plt.scatter(data["Popularity"], data["Price"],c=data['Price'], cmap='autumn')
plt.xlabel("Popularity")
plt.ylabel("Price")

plt.show()

"""These subplots show that HP is the most correlated feature with price."""

#plotting the scatter plot between MPG-H and MPG-C
plt.figure(figsize=(8,4))
plt.title("MPG-H vs MPG-C")
plt.scatter(data["MPG-H"], data['MPG-C'],c=data['MPG-C'], cmap='cool')
plt.xlabel("MPG-H")
plt.ylabel("MPG-C")
plt.show()

"""This graph shows how MPG-H and MPG-C are strongly correlated, as we already observed its high correlation value in the correlation matrix.

Let's analyize the relation between Engine Cylinders, Engine HP, Price.
"""

#Correlation between Cylinders and HP.
plt.figure(figsize=(15,5))
plt.scatter(x=data['Cylinders'], y=data['HP'], color='red', alpha=0.7)
plt.title('Cylinders on HP', weight='bold', fontsize=18)
plt.xlabel('Cylinders', weight='bold',fontsize=14)
plt.ylabel('HP', weight='bold', fontsize=14)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.show()

"""The above graph proves that Cylinders is positively correlated with HP. The cars having high HP values are having more cylinders in general."""

#Correlation between HP and Price of car.
plt.figure(figsize=(15,5))
sns.regplot(x=data['HP'], y=data['Price'], color='red')
plt.title('HP on Price', weight='bold', fontsize=18)
plt.xlabel('Price', weight='bold',fontsize=14)
plt.ylabel('HP', weight='bold', fontsize=14)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.show()

"""HP and Price are highly correlated. Thus more is the HP value, the more is the price of the car."""

#Correlation between Cylinders and Price of car.
plt.figure(figsize=(15,5))
sns.regplot(x=data['Cylinders'], y=data['Price'], color='red')
plt.title('Cylinders on Price', weight='bold', fontsize=18)
plt.xlabel('Price', weight='bold',fontsize=14)
plt.ylabel('Cylinders', weight='bold', fontsize=14)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.show()

"""Cylinders and Price are somewhat correlated with each other, though not too much as we can see in the graph above.

Inference : An engine with more cylinders produces more power, and more power means a high Price.

**Correlation between MPG and Engine HP**
"""

#Correlation between MPG-H and HP..
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
sns.regplot(x=data["HP"], y=data["MPG-H"], line_kws={"color":"red","alpha":1,"lw":5})
plt.title(' MPG-H and HP', weight='bold', fontsize=18)
plt.xlabel('HP', weight='bold',fontsize=14)
plt.ylabel('MPG-H', weight='bold', fontsize=14)
plt.xticks(weight='bold')
plt.yticks(weight='bold')

#Correlation between HP and MPG-C.
plt.subplot(1,2,2)
sns.regplot(x=data["HP"], y=data["MPG-C"], line_kws={"color":"red","alpha":1,"lw":5})
plt.title('MPG-C and HP', weight='bold', fontsize=18)
plt.xlabel('HP', weight='bold',fontsize=14)
plt.ylabel('MPG-C', weight='bold', fontsize=14)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.plot()

"""**Insights**

1.MPH-H and HP are negatively correlated with each other. Hence, we can say that more the MPG-H value, the lesser is the HP value.

2.MPG-C and HP are negatively correlated with each other, though it is not a very strong correlation because many points are scattered and are not following a particular pattern.
"""

print(data.columns)

plt.figure(figsize=(20,8))


# #Checking which kind of transmission is more in demand.
# plt.subplot(1,3,1)
# plt.title("Transmission")
# sns.countplot(x="Transmission", data=data, palette='husl')

plt.subplot(1,3,2)
plt.title("Drive Mode")
sns.countplot(x="Drive Mode", data=data, palette='husl')

plt.subplot(1,3,3)
plt.title("Vehicle Size")
sns.countplot(x="Vehicle Size", data=data, palette='husl')
plt.show()

plt.figure(figsize=(16,10))
plt.title("Vehicle Style")
sns.countplot(y="Vehicle Style", data=data ,hue='Vehicle Size', palette='bright')
plt.xlabel('Count of Vehicles according to Vehicle Size')
plt.ylabel('Vehicle Style')
plt.show()

"""**Insights:**

1.Transmission type: "Automatic", is highest in selling, followed by "Manual".

2.Drive mode: "front wheel drive" is the highest selling. The others are at almost the same selling level.

3.Vehicle size: "Compact" and "Midsize" are more sold as compared to "Large"

4."Sedan- Midsize" and "4dr SUV-Midsize" are the highest selling cars.

5."CargoMinivan", "Cargo Van", "Convertible SUV" and "MiniVan" are the lowest selling cars.
"""

#This is how precisely the Price of cars is distributed, since it could be observed through boxplot properly.
plt.figure(figsize=(16,5))
data["Price"].plot.hist(bins=80, color='purple')
plt.title("Distribution of Price")
plt.xlabel('Price Range')
plt.ylabel('Count')
plt.show()

"""Most number of cars were sold with price range between 2000-3000 range."""

#Having a look at the dataset
data.head(3)

"""# 9.Splitting the dataset into 80 and 20 ratio and building a machine learning model with price as the target variable

Separating X(Predictor Variables) and Y(Prediction Variable) from the dataset.
"""

#Splitting dataset into X and Y values.
X = data[['Year', 'HP', 'Cylinders', 'MPG-H', 'MPG-C','Popularity']].values
y = data['Price'].values

"""Feature Scaling or Standardization: It is a step of Data Pre Processing which is applied to independent variables or features of data. It basically helps to normalise the data within a particular range. Sometimes, it also helps in speeding up the calculations in an algorithm.
Some common methods to perform Feature Scaling are:
1. Min-Max Scaler

2. Standard Scaler

3. Robust Scaler

We will be using Standard scaling technique:
"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaleX = StandardScaler()
scaleY = StandardScaler()

X = scaleX.fit_transform(X)
y = scaleY.fit_transform(y.reshape(-1,1))

"""Splitting the X and Y values into training and testing values.
Usually we keep the test size around 20-30 percent of the dataset. The train set is used to train the machine on the respective algorithm, while the test set is used to predict the values.
"""

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""# 10.Trying different algorithms and checking their performance over metrics.

The algorithms i will be using to predict the Price are:

1. Linear Regression

2. Polynomial Regression

3. Decision Tree Regressor

4. Support Vector Regressor

5. Random Forest Regressor

## 10.1 Linear Regression Model
"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting the Test set results
y_pred = model.predict(X_test)
plt.title("Predictions of Linear Regression model")
plt.xlabel("Y test vales")
plt.ylabel("Y Predictions")
plt.scatter(y_test,y_pred, color='orange')
plt.show()

#Evaluating the model using the following metrics.
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 Score:', metrics.r2_score(y_test, y_pred))

"""## 10.2 Polynomial Regression Model"""

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
model = PolynomialFeatures(degree = 4)
XP = model.fit_transform(X_train)
model.fit(XP, y_train)
reg = LinearRegression()
reg.fit(XP, y_train)

# Predicting a new result with Polynomial Regression
y_pred=reg.predict(model.fit_transform(X_test))
plt.title("Predictions of Polynomial Regression model")
plt.xlabel("Y test vales")
plt.ylabel("Y Predictions")
plt.scatter(y_test,y_pred, color='orange')
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 Score:', metrics.r2_score(y_test, y_pred))

"""## 10.3 Decision Tree Regressor"""

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state = 0)
model.fit(X_train, y_train)

y_test=model.predict(X_test)
plt.title("Predictions of Decision Tree Regression model")
plt.xlabel("Y test vales")
plt.ylabel("Y Predictions")
plt.scatter(y_test,y_pred, color='orange')
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 Score:', metrics.r2_score(y_test, y_pred))

"""## 10.4 Support Vector Regressor"""

#Using SVR model to fit our dataset
from sklearn.svm import SVR
model = SVR(kernel = 'rbf')
model.fit(X_train, y_train)

# Predicting a new result
y_pred = model.predict(X_test)
plt.title("Predictions of SVR Regression model")
plt.xlabel("Y test vales")
plt.ylabel("Y Predictions")
plt.scatter(y_test,y_pred, color='orange')
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 Score:', metrics.r2_score(y_test, y_pred))

"""## 10.5 Random Forest Regressor"""

#Random Forest Regressior for the dataset
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 200, random_state = 0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
plt.title("Predictions of Random Forest Regression model")
plt.xlabel("Y test vales")
plt.ylabel("Y Predictions")
plt.scatter(y_test,y_pred, color='orange')
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 Score:', metrics.r2_score(y_test, y_pred))