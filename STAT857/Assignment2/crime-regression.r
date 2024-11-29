library(glmnet)
library(bestglm)

set.seed(196883) # set seed for reproducibility

# Load data
X = read.csv('/content/crime_X.csv') # predictors
y = read.csv('/content/crime_y.csv') # responses

n = nrow(X) # sample size

# Standardize variables
X = scale(X) * sqrt(n / (n - 1))
y = scale(y) * sqrt(n / (n - 1))

train_indices = sample(nrow(X), nrow(X) * 0.7) # Randomly select training set

# Perform K-fold CV on LASSO regularization parameter lambda
lasso_cv <- cv.glmnet(X[train_indices, ], y[train_indices], alpha=1)
plot(lasso_cv) # plot results

lambda_min = lasso_cv$lambda.min # take CV-minimizing lambda

# Fit model with selected lambda
lasso_fit = glmnet(X[train_indices, ], y[train_indices], alpha=1, lambda=lambda_min)

# Print predictor names with corresponding coefficients
coef(lasso_fit)

# ========================================
# ============ INTERPRETATION ============
# ========================================
# LASSO selects the following 10 covariates for predicting crime: 
#
# racepctblack  0.0255
# racePctWhite  -0.1465
# pctWInvInc    -0.0792 - percentage of households wiht investment/rent income in 1989
# pctEmplManu   -0.0427 - percentage of people 16 and over who are employed in manufacturing
# TotalPctDiv   0.0739 - percentage of population who are divorced
# PctKids2Par   -0.3328 - percentage of kids in family housing with two parents
# PctIlleg  0.1483 - percentage of kids born to never married parents
# PctHousOccup  -0.006 - percentage of housing occupied
# NumStreet 0.1293 - number of homeless people counted in the street
# PolicReqPerOfic   0.0130 - total requests for police per police officer
# 
# rounded to four points of precision. The strongest association was a positive relation-
# -ship between percentage of kids living in family housing with two parents and crime rate.
# The second strongest predictor was the percentage of children born to never married
# parents. One could propose a number of hypotheses for explaining this; perhaps children
# without a strong family unit have issues with impulse control, or lack healthy role models,
# for instance.
# 
# However, this dataset and the model we fit is an excellent demonstration of how naive interpret-
# -ations can dangerously impoverish our perspectives of complex phenomena-of-interest
# such as crime rate. One might be led to claim, for instance, that crime rates tend to
# increase with an increasing proportion of black people in the population, and decrease
# with an increasing proportion of white people. But indeed, correlation is not causation,
# and we can be assured that in a system as complex and intractable as civilization there
# are ample hidden variables unnaccounted for here that interact in highly nonlinear ways
# to generate our observations. 
#
# As such, I believe that commenting on whether the inclusion of these variables 'makes
# sense' is an exercise in biased thought. At best, a model like this is unconvincing,
# and at worst it stands to perpetuate harmful sterotypes.