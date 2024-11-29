# ###############################################
# ########## README #############################
# ###############################################
# - The code will not run unless all dependencies
#   (as indicated in the IMPORTS section) are
#   installed.
# - In the MAIN FUNCTION, please modify the
#   path variables in the CONTROL PANEL to point
#   to the location of the 'train.csv' and 
#   'test.csv' files on your system. The rest of
#   the code will run without modification.
# - Runtime is roughly 15-20m to train the model.
# ###############################################

# ========== IMPORTS ==============================
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline

import tensorflow as tf

warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================================
# ========== HELPER FUNCTIONS ==============================
# ==========================================================

def prep_submission(test_dataframe, pred, path):
    '''Organize predictions into the competition-specified format for submission'''
    output = pd.concat([test_dataframe['ID'], pd.DataFrame(pred)], axis=1)
    output.columns = ['ID', 'pred']

    output.to_csv(path, index=False)

def sparse_batch_generator(X, y, batch_size):
    '''Convert sparse design matrices into dense batch arrays for passage through a TensorFlow DNN'''

    num_batches = X.shape[0] / batch_size
    k = 0

    shuffle_ind = np.arange(y.shape[0])
    np.random.shuffle(shuffle_ind)
    X =  X[shuffle_ind, :]
    y =  y[shuffle_ind]

    while 1:
        batch_ind = shuffle_ind[(batch_size * k):(batch_size * (k + 1))]
        X_batch = X[batch_ind, :].todense()
        y_batch = y[batch_ind]
        k += 1

        yield(np.array(X_batch), y_batch)

        if k >= num_batches:
            np.random.shuffle(shuffle_ind)
            k=0

# =======================================================
# ========== MAIN FUNCTION ==============================
# =======================================================

def main():

    # ##############################################################
    # ########## CONTROL PANEL #####################################
    # ##############################################################
    data_train = pd.read_csv('./train.csv') # path to training data
    data_test = pd.read_csv('./test.csv') # path to testing data
    # ##############################################################

    # Combine train/test data to fit imputers and transformers
    data_merged = pd.concat([data_train, data_test], ignore_index=True, sort=False) 

    # ==============================================================
    # ========== CLASSIFYING FEATURES ============================== 
    # ==============================================================

    # Object of regression
    TARGET = ['log_price'] 

    # Predictors taking value in an unordered, discrete set
    NOMINAL = ['property_type', 'room_type', 'bed_type', 'cancellation_policy', 'cleaning_fee', 'city', 'host_has_profile_pic', 'host_identity_verified',
            'instant_bookable']

    # Predictors taking value in an ordered, discrete set
    ORDINAL = ['accommodates', 'bathrooms', 'number_of_reviews', 'bedrooms', 'beds']

    # Predictors taking value on a continuum
    NUMERICAL = ['latitude', 'longitude', 'review_scores_rating']

    # Natural language predictors
    TEXT = ['amenities', 'description', 'name']

    # Exclude predictors that present as dates, or that I otherwise deem unimportant
    EXCLUDE = ['first_review', 'host_response_rate', 'host_since', 'last_review', 'zipcode', 'ID', 'neighbourhood']

    FEATURES = NOMINAL + ORDINAL + NUMERICAL + TEXT # combine the lists 

    # ===================================================================
    # ========== IMPUTATION/TRANSFORMATION ============================== 
    # ===================================================================

    ordinal_preproc = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), # replace NaN values with the most frequent entry
        ('encoder', OrdinalEncoder()) # onehot encoding 
    ])

    nominal_preproc = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), # replace NaN values with the most frequent entry
        ('encoder', OneHotEncoder(sparse_output=True, handle_unknown='ignore')) # onehot encoding, do not alter unfitted classes
    ])

    numerical_preproc = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')), # replace NaN values with the mean value 
        ('scaler', StandardScaler()) # normalize data
    ])

    text_preproc = Pipeline([
        ('vectorizer', TfidfVectorizer()) # term frequency-inverse document frequency vector encoding
    ])

    # Combined preprocessing pipeline which acts directly on the pandas dataframes
    preproc_pipe = ColumnTransformer([
        ('nominal_preprocessor', nominal_preproc, NOMINAL),
        ('ordinal_preprocessor', ordinal_preproc, ORDINAL),
        ('numerical_preprocessor', numerical_preproc, NUMERICAL),
        *[('text_preprocesser_{}'.format(i), text_preproc, col) for i, col in enumerate(TEXT)]
    ])

    # Leave target quantity transformer separate so we can more easily invert the standardization later 
    preproc_target = ColumnTransformer([
        ('target_preprocessor', numerical_preproc, TARGET)
    ])

    # Fit the preprocessors to the combined data to avoid extrapolation during testing
    preproc_target.fit(data_merged)
    preproc_pipe.fit(data_merged)

    # =====================================================================
    # ========== APPLY PREPROCESSING TO DATA ============================== 
    # =====================================================================

    seed = 196883 # set state for generating train-val splits
    val_prop = 0.2 # proportion of training data to preserve for validation

    # Preprocess training data
    X_train = preproc_pipe.transform(data_train[FEATURES])
    y_train = preproc_target.transform(data_train[TARGET])

    # Split data into training/validation subsets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=seed, train_size=(1 - val_prop), shuffle=True)

    # Preprocess test data using the transformer fit to the combined dataset
    X_test = preproc_pipe.transform(data_test[FEATURES])

    # ===============================================================
    # ========== MULTILAYER PERCEPTRON ============================== 
    # ===============================================================

    tf.keras.utils.set_random_seed(
        196883 # random seed for reproducibility
    )

    # Training hyperparameters
    BATCH_SIZE = 30
    NUM_EPOCHS = 31
    STEPS_PER_EPOCH = X_train.shape[0] // BATCH_SIZE

    VAL_BATCH_SIZE = BATCH_SIZE
    VALIDATION_STEPS = X_val.shape[0] // VAL_BATCH_SIZE

    lr=1e-4

    # Build the network
    bnb_regnet = tf.keras.models.Sequential() # feedforward neural network

    # Input layer
    bnb_regnet.add(tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],))) 

    # Alternating dense/dropout layers
    bnb_regnet.add(tf.keras.layers.Dense(64, activation='relu', kernel_initializer='normal', input_shape=(X_train.shape[1],)))
    bnb_regnet.add(tf.keras.layers.Dropout(0.6))
    bnb_regnet.add(tf.keras.layers.Dense(32, activation='relu', kernel_initializer='normal'))
    bnb_regnet.add(tf.keras.layers.Dropout(0.4))
    bnb_regnet.add(tf.keras.layers.Dense(32, activation='relu', kernel_initializer='normal'))
    bnb_regnet.add(tf.keras.layers.Dropout(0.3))
    bnb_regnet.add(tf.keras.layers.Dense(32, activation='relu', kernel_initializer='normal'))
    bnb_regnet.add(tf.keras.layers.Dropout(0.3))
    bnb_regnet.add(tf.keras.layers.Dense(16, activation='relu', kernel_initializer='normal'))

    # Ouput layer
    bnb_regnet.add(tf.keras.layers.Dense(1))

    # Compile the network with MSE loss and Adam optimizer
    bnb_regnet.compile(
        loss = tf.keras.losses.MeanSquaredError(),
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    )

    # Print model architecture to console
    bnb_regnet.summary()

    # Fit the model
    history = bnb_regnet.fit(
        x=sparse_batch_generator(X_train, y_train, BATCH_SIZE),
        epochs=NUM_EPOCHS,
        validation_data=sparse_batch_generator(X_val, y_val, 1),
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        verbose=1
        )

    # ===============================================================
    # ========== PREDICTION AND OUTPUT ============================== 
    # ===============================================================

    # Form predictions over the test data using the fitted MLP
    pred = bnb_regnet.predict(X_test)

    # Unstandardize predictions using fitted transformer
    pred = preproc_target.transformers_[0][1][1].inverse_transform(pred.reshape((-1, 1)))

    # Ouput name and path destination
    filename = 'contest_submission.csv'
    output_path = './{}'.format(filename)

    # Package and save submission file to the specified directory
    prep_submission(data_test, pred, output_path)

if __name__ == '__main__':
    main()