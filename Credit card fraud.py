# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 17:15:22 2022

@author: Alex
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split


# open dataset
df = pd.read_csv("Credit card data.csv")



# get statistical data of the "amount column" 
df['amount'].describe()



# get statistical data of the "type" column 
df['type'].info()



# 3.create new column and assign values (1 for debit or payment, and 0 for otherwise) 

# Method 1
# this goes back to encoding categorical variables
# I will use ordinal encoding for this, using a dictionary of key-value pairs
# once the dict is created it can be mapped onto the desired column
isPayment1 = {'PAYMENT': 1, 'DEBIT': 1, 'CASH_OUT': 0, 'TRANSFER': 0, 'CASH_IN': 0}
df['isPayment'] = df['type'].map(isPayment1)



# Method 2 (Ric)
# Add new column with vales 0 
# Add value 1 when the type is payment or debit
df['isPayment'] = 0
df.loc[df['type'] == "PAYMENT", 'isPayment'] = 1
df.loc[df['type'] == "DEBIT", 'isPayment'] = 1



# 4. create an column call ismovement with 1 values for CASH OUT and transfer
df['ismovement'] = 0
df.loc[df['type'] == "CASH OUT", 'ismovement'] = 1
df.loc[df['type'] == "TRANSFER", 'ismovement'] = 1



# 5. create a column with the absolute difference between oldbalanceorg and oldbalanceDest
df['accountDiff'] = 0
df['accountDiff'] = df['oldbalanceOrg'] - df['oldbalanceDest']
print(df.info())



# 6.Create a variable called features which will be an array consisting of 
#the following fields:amount, isPayment, isMovement, accountDiff
features = df[['amount', 'isPayment','ismovement', 'accountDiff']]



# 7. Split the data using train_test_split
# 7.1 First, extract the data from the column is fraud and store it into y for the model
y = df['isFraud']
# Use a test_size value of 0.3
features_train, features_test, y_train, y_test = train_test_split(features, y, train_size=0.7, test_size=0.3, random_state=60)



# 8.Create a StandardScaler object, 
# 8.1.fit_transform() it on the training features, and .transform() the test features.
scaler = StandardScaler()
x_train = scaler.fit_transform(features_train)
x_test = scaler.fit_transform(features_test)



# 9. Create and LR model and .fit() it on the training data
model = LogisticRegression()
model.fit(x_train, y_train)



# 10. Run the model's score method on the training data and print the score
# 10.1 Store intercept and coeficient
intercept = model.intercept_
coef = model.coef_


# 10.2 Calculate log odds
log_odds = intercept + coef * x_train
print(log_odds)


# 10.3 Calculate predicted fraudulence 
y_train_predicted = model.predict(x_train)
print(y_train_predicted)


# 10.4 Confusion matrix
from sklearn.metrics import confusion_matrix
y_train = y_train_predicted.reshape(-1,1)

confusion = (confusion_matrix(y_train, y_train_predicted))
print(confusion)

#10.5 Calculate Scores
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

accuracy_model = accuracy_score(y_train, y_train_predicted)
precision_model = precision_score(y_train, y_train_predicted)
recall_model = recall_score(y_train, y_train_predicted)
f1_model = f1_score(y_train, y_train_predicted)

print(accuracy_model, precision_model,recall_model,f1_model ) 



# 11. Run the model's score method on the test data and print the score


# 11.1 Calculate log odds
log_odds_test = intercept + coef * x_test
print(log_odds_test)


# 11.2 Calculate predicted fraudulence 
y_test_predicted = model.predict(x_test)
print(y_test_predicted)


# 11.3 Confusion matrix
from sklearn.metrics import confusion_matrix
y_test_predicted = y_test_predicted.reshape(-1,1)

confusion = (confusion_matrix(y_test, y_test_predicted))
print(confusion)

#11.4 Calculate Scores
accuracy_model_test = accuracy_score(y_test, y_test_predicted)
precision_model_test  = precision_score(y_test, y_test_predicted)
recall_model_test  = recall_score(y_test, y_test_predicted)
f1_model_test  = f1_score(y_test, y_test_predicted)

print(accuracy_model_test , precision_model_test ,recall_model_test , f1_model_test  ) 


#12 Print the coefficients for our model
print(coef)


#13 Predict with the model
# There are three numpy arrays pre-loaded in the workspace
# Create a fourth array, your_transaction, and add any transaction information you’d like
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])
your_transaction = np.array([49835.22, 0.0, 1.0, 0.5])


#14 Combine the new transactions and your_transaction into a single numpy array called sample_transactions
sample_transactions = np.array([transaction1, transaction2, transaction3, your_transaction])

#15 We must also scale the feature data we are making predictions on
sample_transactions = scaler.fit_transform(sample_transactions)


# Call your model’s .predict_proba() method on sample_transactions and print the result.
prediction = model.predict(sample_transactions)
probability = model.predict_proba(sample_transactions)
print(probability)
