from input_processing import read_and_preprocess
from NaiveBayesClassifier import NaiveBayesClassifier

# feature names, label name of the input files
COLUMN_NAMES = [
    'age', 'gender', 'height_cm', 'weight_kg', 'body fat_%',
    'diastolic', 'systolic', 'grip_force', 'sit_and_bend_forward_cm',
    'sit_up_count', 'broad_jump_cm', 'class']
LABEL_NAME = 'class'

if __name__ == '__main__':
    # Read input data
    x_train, y_train, x_test, y_test = read_and_preprocess(column_names=COLUMN_NAMES, label_name=LABEL_NAME)

    # Create classifier
    nbc = NaiveBayesClassifier(feature_names=COLUMN_NAMES, plotting=False, logging=False)
    nbc.train(x_train, y_train)
    nbc.classify(x_test, y_test)
