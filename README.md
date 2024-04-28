# Module 21 Deep-Learning-Model Analysis

![Screenshot 2024-04-28 at 1 21 33 PM](https://github.com/apkaur32/deep-learning-challenge/assets/150749167/f63e10ca-7ba7-4503-9e81-0b7bb99c58be)

## Overview of the analysis: 
The purpose of this analysis was to use machine learning and neural networks to assess the features in the provided dataset in order to create a binary classifier that can predict whether applicants will be successful if funded by nonprofit foundation named Alphabet Soup. The CSV contains more than 34,000 organizations that have received funding from Alphabet Soup over the years and our goal is to design a deep learning model which makes an accurate forecast for application success rate.

## Results:

> Data Preprocessing

1.What variable(s) are the target(s) for your model?/
y = 'IS_SUCCESSFUL' is the target variable. 

2.What variable(s) are the features for your model?/
X = all the remaining variables are the features from the application_df when ".drop(columns="IS_SUCCESSFUL")" is applied: 'NAME','APPLICATION_TYPE','AFFILIATION','CLASSIFICATION','USE_CASE','ORGANIZATION','STATUS','INCOME_AMT','SPECIAL_CONSIDERATIONS','ASK_AMT.

3.What variable(s) should be removed from the input data because they are neither targets nor features?/
'EIN' is absolutely unique identifier and should be removed from input data. 

> Compiling, Training, and Evaluating the Model

4.How many neurons, layers, and activation functions did you select for your neural network model, and why?/
5.Were you able to achieve the target model performance?/ 
6.What steps did you take in your attempts to increase model performance?/

I took the following steps to optimize the model:/

-Attempt #1: Add more neurons
1st hidden layer: units=120, activation function="relu"
2nd hidden layer: units=60, activation function="relu"
RESULTS 1: Accuracy: 72.6%
Adding more neurons did not increase accuracy, tried different activation functions next.

-Attempt #2: Add another hidden layer, use different activation functions, reduce epochs number
1st hidden layer: units=120, activation function="relu"
2nd hidden layer: units=60, activation function="relu"
3rd hidden layer: units=20, activation function="tanh"
number of epochs=50
RESULTS 2: Accuracy: 72.4%
Adding more hidden layer, and using different activation function still did not increase any accuracy. Tried another combination next.

-Attempt #3: Add Dropout to minimize co-adaptation of nodes and reduce overfitting; and use activation function: LeakyReLU
1st hidden layer: units=60, activation function="LeakyReLU", Dropout(0.2)
2nd hidden layer: units=20, activation function="LeakyReLU", Dropout(0.2)
number of epochs=50
RESULTS 3: Accuracy: 72.5%
No improvement in acccuracy so I revisited the preprocessed dataset and decided to drop fewer columns next. 

-Attempt #4: (FINAL) by keeping 'NAME' column intact and creating bins
1st hidden layer: units=80, activation function="relu"
2nd hidden layer: units=30, activation function="relu"
RESULTS 4: ~ 78% Accuracy achieved
Not dropping 'NAME' column from the dataset and creating bins in this column for rare values fewer than 10 was the right approach for optimizing our model.

## Summary: overall results and recommendation
The overall results of the deep learning model predict that applicants whose: 
'NAME' appears 10 times or more;
'APPLICANT_TYPE' is either: T3,T4,T6,T5,T19,T8,T7,T10; 
'CLASSIFICATION' is either: C1000,C2000,C1200,C3000,C2100;  
have the best chance of success in their ventures, with the current accuracy rate of ~78% using our model. This also meant that evaluating the 'NAME' column produces a higher success rate than our original trained model at ~72% and should be kept in the dataset for retraining. 

Another recommendation for a different model that could solve this classification problem could be optimizing our current model further to include more layers and add more neurons, which seems to improve accuracy to ~80% in some instances, though I did not include them in my analysis directly. 
