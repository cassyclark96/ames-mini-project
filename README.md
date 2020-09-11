# Ames Housing Project

## Objective

The objective of this project was to explore the Ames Housing dataset and uses different regression models in order to predict the sale price of homes in Ames, Iowa.

## Requirements

- Jupyter notebook, python 3

## Imports

For this project, I imported a lot of python libraries which included: 
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
```

## Table of Contents

1) Datasets
    - Contains Kaggle submissions datasets, original training and testing datasets, as well as the cleaned up versions of the training and testing datasets.
    - A description of the original dataset can be found here http://jse.amstat.org/v19n3/decock/DataDocumentation.txt
2) EDA and Data Cleaning notebook
3) Kaggle Model Submissions notebook
4) Modeling notebook

## Summary

I was given a training dataset of 2051 rows and 81 columns and a testing dataset with 878 rows and 80 columns to work with in building models to use against the test dataset in order to predict the housing prices. The cleaning steps I took included converting all the ordinal data into a numerical ranking system. I also used dummy variables for neighborhoods and garage types. I made mini data frames to compare various aspects to price in order to figure out which variables had the highest correlation to sale price. The cleaned data sets contained 2051 rows and 121 columns for the train set and 878 rows and 119 columns for the test set.

Before beginning the modeling process, a baseline score was found of roughly $79,239 by finding the mean of the saleprice in the training set and then running the RMSE equation on the mean and actual sale prices from training data. The models tested in this project include a linear regression, ridge, and lasso model. The first round of models were run with mostly the original columns. After I finished running the first few models to get an idea where the data stood in it's current state against the baseline, I began playing around with feature engineering and used my findings from the mini data frames I made. The best model I came up with ended up with an RMSE score of roughly $27,599. This model was a standard linear regression model that included 26 features.

## Next Steps

The next steps for this project would be to find any outliers I may have missed as well as test out different feature combinations to see I can imporve the RMSE score.
