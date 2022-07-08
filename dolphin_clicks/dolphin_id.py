"""
Jaylan Pierce
November 23, 2021
"""
# Makes life easier for appending to lists
from collections import defaultdict
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Third party libraries
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

#For confusion matrix
import matplotlib.pyplot as plt

#My libraries
from lib.partition import split_by_day
import lib.file_utilities as util

def get_sets(data_directory):
    """
    Importing data and putting it into a dictionary which is keyed by date
    Splitting the data into train and test data
    Preparing Data to Train Model
    """

    "Importing data and putting it into a dictionary which is keyed by date"
    input_data = util.get_files(data_directory, stop_after=100)
    parsed_data = util.parse_files(input_data)
    group_by_date = split_by_day(parsed_data)
    list_of_days = list(group_by_date.keys())


    "Splitting the data into train and test data"
    X_train, X_test = train_test_split(list_of_days)

    "Getting Features"
    ######TRAINING FEATURES######
    shape = (0, 20)
    features_train_data = np.empty(shape)
    for day in X_train:
        for i in range(len(group_by_date[day])):
            features_train_data = np.vstack((features_train_data, getattr(group_by_date[day][i], 'features')))
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    ######TESTING FEATURES#######
    shape = (0, 20)
    features_test_data = np.empty(shape)
    for day in X_test:
        for i in range(len(group_by_date[day])):
            features_test_data = np.vstack((features_test_data, getattr(group_by_date[day][i], 'features')))

    if getattr(group_by_date[day][i], 'label') == 'Gg':
        fill_value = 0
    else:
        fill_value = 1

    label_train_data = np.full(features_train_data.shape[0], fill_value=fill_value)
    label_test_data = np.full(features_test_data.shape[0], fill_value=fill_value)


    return [label_train_data, label_test_data, features_train_data, features_test_data]



def dolphin_classifier(Gg_data, Lo_data):
    """
    Neural net classification of dolphin echolocation clicks to species
    :param data_directory:  root directory of data
    :return:  None
    """
    plt.ion()   # enable interactive plotting
    use_onlyN = np.Inf  # debug, only read this many files for each species

    x_train = np.concatenate((Gg_data[2], Lo_data[2]))
    x_test = np.concatenate((Gg_data[3], Lo_data[3]))
    y_train = np.concatenate((Gg_data[0], Lo_data[0]))
    y_test = np.concatenate((Gg_data[1], Lo_data[1]))

    "Building Model"
    model = Sequential()

    model.add(Input(shape=(20,)))
    model.add(Dense(10, activation='relu', activity_regularizer='l2'))
    model.add(Dense(10, activation='relu', activity_regularizer='l2'))
    model.add(Dense(2, activation='softmax'))

    "Training Model"
    model.compile(optimizer='Adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train,
              y_train,
              batch_size=100,
              validation_data=(x_train, y_train),
              epochs=10,
              verbose=True,
              validation_split=.01)
    results = model.evaluate(x_test, y_test)
    print(results)



if __name__ == "__main__":

    Gg_data_directory = "/Users/jaylanpierce/Desktop/A4/features/Gg"
    Gg_data = get_sets(Gg_data_directory)
    Lo_data_directory = "/Users/jaylanpierce/Desktop/A4/features/Lo"
    Lo_data = get_sets(Lo_data_directory)

    dolphin_classifier(Gg_data, Lo_data)
