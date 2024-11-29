# Imports
library(e1071)

# Print working directory
getwd()

# Import train and test data
train = read.csv('./STAT857/Assignment4/data/asg4_train_data.csv')
test = read.csv('./STAT857/Assignment4/data/asg4_test_data.csv')

# Examine the dataframe
head(train)

# Encode the target variable as a factor
train$diabetes = as.factor(train$diabetes)
test$diabetes = as.factor(test$diabetes)

# Fit the SVM with a radial basis kernel
model = svm(diabetes~., data=train, kernel='radial', cost=1, type='C-classification')

# Make prediction on the test data
svm_pred = predict(model, test)

# Form the confusion matrix
confusion_matrix = table(svm_pred, test$diabetes)
confusion_matrix # Print the confusion matrix to the console

# Compute the prediction accuracy
svm_prop_corr_pred = sum(diag(confusion_matrix)) / sum(confusion_matrix)
svm_prop_corr_pred # Print prediction accuracy

# Print the normalized confusion matrix
confusion_matrix / sum(confusion_matrix)

# =========================================
# =============== RESULTS =================
# =========================================

# Confusion matrix (nornalized)
# svm_pred        neg        pos
#      neg 0.59322034 0.17796610
#      pos 0.07627119 0.1525423

# Confusion matrix (un-normalized)
# svm_pred neg pos
#      neg  70  21
#      pos   9  18

# Prediction Accuracy (~74%):
# 0.7457627 