# B211--Assignment-5

PART 2: EXPLANATION: 
The logistic regression model seemed the best fit for the data. Logistic regression is able to be the best due to how it can handle binary classification problems and can model the relationship between the features and the cancer diagnosis. The K neighbors model also preformed relatively well but was less accurate than logisitc regression since it depends on the distance between data points. Finally, the decision tree model had the lowest accuracy because individual trees can sometimes overfit the training data which reduces their preformance on new data. 

README FILE:

Purpose: The purpose of this assingment is to start performing machine learning classification using Python and the Scikit Learn module, using real world data.

Class Design/Implementation:
X(Features): Numerical measurement describing the characteristics of the tumor.
Y (labels): Classification values (when its 0 it is malignant, when 1 it is benign).
train_test_split: divides dataset into training and testing data.
fit: trains the model.
predict: Makes predictions on test data.

Limitations: 
1. The dataset is pretty small.
2. Accuracy cant tell us everything about the dataset.
