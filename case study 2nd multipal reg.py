# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:32:58 2021

@author: Sunil Chintale
"""
#Import Libraries
import pandas as pd

#Import data
dataset = pd.read_csv("c:/tableau/T6_Luxury_Cars.csv")

#Identifying/Finding missing values if any----
dataset.isnull()
dataset.isnull().sum()

dataset.info()


pd.get_dummies(dataset["DriveTrain"])
pd.get_dummies(dataset["DriveTrain"],drop_first=True)
DriveTrain_Dummy = pd.get_dummies(dataset["DriveTrain"],drop_first=True)
DriveTrain_Dummy.head(5) 



pd.get_dummies(dataset["Origin"])
pd.get_dummies(dataset["Origin"],drop_first=True)
Origin_Dummy = pd.get_dummies(dataset["Origin"],drop_first=True)
Origin_Dummy.head(5) 


pd.get_dummies(dataset["Make"])
pd.get_dummies(dataset["Make"],drop_first=True)
make_Dummy = pd.get_dummies(dataset["Make"],drop_first=True)
make_Dummy.head(5) 



pd.get_dummies(dataset["Type"])
pd.get_dummies(dataset["Type"],drop_first=True)
type_Dummy = pd.get_dummies(dataset["Type"],drop_first=True)
type_Dummy.head(5) 



dataset = pd.concat([dataset,DriveTrain_Dummy,make_Dummy,type_Dummy,Origin_Dummy],axis=1)
dataset.head(5)
#dropping the columns whose dummy var have been created
dataset.drop(["Make","Model","Origin","Type","DriveTrain",],axis=1,inplace=True)
dataset.head(5)
#Obtaining DV & IV from the dataset

X=dataset.drop("MPG (Mileage)",axis=1)
y=dataset["MPG (Mileage)"]

    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# Accuracy of the model

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


regressor.coef_

# Intercept
regressor.intercept_


# this models gives us 83% accuracy
         
  # backward elimination for incrasing accuracy
  #Importing the library:
import statsmodels.api as sm

#Adding a column in matrix of features:
import numpy as nm
X = nm.append(arr = nm.ones((426,1)).astype(int), values=X, axis=1)

#Applying backward elimination process now
#Firstly we will create a new feature vector x_opt, which will only contain a set of 
#independent features that are significantly affecting the dependent variable.
x_opt=X[:, [ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52]]

#for fitting the model, we will create a regressor_OLS object of new class OLS of 
#statsmodels library. Then we will fit it by using the fit() method.
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()

#We will use summary() method to get the summary table of all the variables.
regressor_OLS.summary()

#In the above summary table, we can clearly see the p-values of all the variables. 
#Here x1, x2 are dummy variables, x3 is R&D spend, x4 is Administration spend, and x5 is Marketing spend.

#Now since x5 has highest p-value greater than 0.05, hence, will remove the x1 variable
#(dummy variable) from the table and will refit the model.

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()


x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,42,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,42,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,42,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()


x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,42,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,42,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,42,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,42,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,42,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,42,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,42,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,42,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,42,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,42,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,42,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,42,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,18,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,18,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,11,18,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,9,10,18,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,8,18,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,6,7,18,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,1,2,3,4,5,7,18,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()


x_opt=X[:,[ 0,1,3,4,7,18,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt=X[:,[ 0,3,4,7,18,46,47,48,49,50,]] 
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()


#Hence,only  DriveTrain,horsepower,weight,wheelbase independent variable is a significant variable for the prediction. 
#So we can now predict efficiently using this variable.

#----------Building Multiple Regression model by only using DriveTrain,horsepower,weight,wheelbase:-----------------

#Extracting Independent and dependent Variables 
# taken dependent variales as same on upper side (i.e = y) and independent variales as ((x_opt)
# Splitting the dataset into training and test set.  

from sklearn.model_selection import train_test_split
x_BE_train, x_BE_test, y_BE_train, y_BE_test= train_test_split(x_opt, y, test_size= 0.25, random_state=0)

#Fitting the MLR model to the training set:  
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_BE_train, y_BE_train)

#Predicting the Test set result;
y_pred= regressor.predict(x_BE_test)

#Cheking the score  
#Calculating the r squared value:
from sklearn.metrics import r2_score
r2_score(y_BE_test,y_pred)
#The above score tells that our model is now more accurate with the test dataset with
#accuracy equal to 85.03%

#Calculating the coefficients:
print(regressor.coef_)

#Calculating the intercept:
print(regressor.intercept_)




















































































































