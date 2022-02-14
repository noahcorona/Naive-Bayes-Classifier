import sys  # For taking in console inputs
from pandas import read_csv as read_csv  # For statistical functions, data manipulation
from timing import timeit


# Helper function to read a csv file
# Assumes the csv has no label info
# Returns tuple (data, labels) of pandas dataframes
def read_and_split_file(file_name, column_names, label_name):
    df = read_csv(file_name,
                  header=0,
                  names=column_names)
    col_names = column_names[0:len(column_names) - 1]
    return df.drop([label_name], axis=1), df.drop(col_names, axis=1)


# Import list of file names given as console args
# Ignores .py to avoid adding self
# Returns a tuple of pandas dataframes
@timeit
def read_files(column_names, label_name):
    # We should only have 2 files here, testing.txt and training.txt
    train_data, train_labels, test_data, test_labels = None, None, None, None
    for i, file_name in enumerate(sys.argv):
        if file_name == 'testing.txt':
            test_data, test_labels = read_and_split_file(file_name=file_name, column_names=column_names,
                                                         label_name=label_name)
        elif file_name == 'training.txt':
            train_data, train_labels = read_and_split_file(file_name=file_name, column_names=column_names,
                                                           label_name=label_name)
    return test_data, test_labels, train_data, train_labels


# Perform all input preprocessing
# Returns a tuple of pandas dataframes
def preprocess(train_data, train_labels, test_data, test_labels):
    train_data = train_data.replace(['M'], 0.0).replace(['F'], 100.0)
    test_data = test_data.replace(['M'], 0.0).replace(['F'], 100.0)
    train_data = (train_data - train_data.mean()) / train_data.std()
    test_data = (test_data - test_data.mean()) / test_data.std()
    return test_data, test_labels, train_data, train_labels


# Function to combine for timing purposes
@timeit
def read_and_preprocess(column_names, label_name):
    train_data, train_labels, test_data, test_labels = read_files(column_names, label_name)
    return preprocess(train_data, train_labels, test_data, test_labels)
