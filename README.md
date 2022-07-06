# Logistic-Regression-predicting-credit-card-fraud-Codeacademy-project-
In this project, you are a Data Scientist working for a credit card company. You have access to a dataset (based on a synthetic financial dataset), that represents a typical set of credit card transactions. Your task is to use Logistic Regression and create a predictive model to determine if a transaction is fraudulent or not.

Project based on a Kaggle Synthetic Financial Datasets For Fraud Detection: https://www.kaggle.com/datasets/ealaxi/paysim1
The code works equally well with the original dataset and the Codacademy modified dataset

1.The file transactions.csv contains data on 200k simulated credit card transactions. Let’s begin by loading the data into a pandas DataFrame named transactions. Print a few rows from transactions and the total row count. What columns do we have already that can help with detecting fraud?

2.Looking at the dataset, combined with our knowledge of credit card transactions in general, we can see that there are a few interesting columns to look at. We know that the amount of a given transaction is going to be important. Calculate summary statistics for this column. What does the distribution look like?

3.We have a lot of information about the type of transaction we are looking at. Let’s create a new column called isPayment that assigns a 1 when type is “PAYMENT” or “DEBIT”, and a 0 otherwise.

4.Similarly, create a column called isMovement, which will capture if money moved out of the origin account. This column will have a value of 1 when type is either “CASH_OUT” or “TRANSFER”, and a 0 otherwise.

5.With financial fraud, another key factor to investigate would be the difference in value between the origin and destination account. Our theory, in this case, being that destination accounts with a significantly different value could be suspect of fraud. Let’s create a column called accountDiff with the absolute difference of the oldbalanceOrg and oldbalanceDest colum

6.
Before we can start training our model, we need to define our features and label columns. Our label column in this dataset is the isFraud field. Create a variable called features which will be an array consisting of the following fields:

amount
isPayment
isMovement
accountDiff
Also create a variable called label with the column isFraud.


7.Split the data into training and test sets using sklearn‘s train_test_split() method. We’ll use the training set to train the model and the test set to evaluate the model. Use a test_size value of 0.3.

8.Since sklearn‘s Logistic Regression implementation uses Regularization, we need to scale our feature data. Create a StandardScaler object, .fit_transform() it on the training features, and .transform() the test features.

9.Create a LogisticRegression model with sklearn and .fit() it on the training data.

Fitting the model find the best coefficients for our selected features so it can more accurately predict our label. We will start with the default threshold of 0.5.


10.Run the model’s .score() method on the training data and print the training score.

Scoring the model on the training data will process the training data through the trained model and will predict which transactions are fraudulent. The score returned is the percentage of correct classifications, or the accuracy.

11.Run the model’s .score() method on the test data and print the test score.

Scoring the model on the test data will process the test data through the trained model and will predict which transactions are fraudulent. The score returned is the percentage of correct classifications, or the accuracy, and will be an indicator for the sucess of your model.

How did you model perform?

12.Print the coefficients for our model to see how important each feature column was for prediction. Which feature was most important? Least important?

13.Let’s use our model to process more transactions that have gone through our systems. There are three numpy arrays pre-loaded in the workspace with information on new sample transactions under “New transaction data”

Create a fourth array, your_transaction, and add any transaction information you’d like. Make sure to enter all values as floats with a .!

14.Combine the new transactions and your_transaction into a single numpy array called sample_transactions.


15.Since our Logistic Regression model was trained on scaled feature data, we must also scale the feature data we are making predictions on. Using the StandardScaler object created earlier, apply its .transform() method to sample_transactions and save the result to sample_transactions.

16.Which transactions are fraudulent? Use your model’s .predict() method on sample_transactions and print the result to find out.

Want to see the probabilities that led to these predictions? Call your model’s .predict_proba() method on sample_transactions and print the result. The 1st column is the probability of a transaction not being fraudulent, and the 2nd column is the probability of a transaction being fraudulent (which was calculated by our model to make the final classification decision).
