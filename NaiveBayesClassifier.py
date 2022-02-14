import sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import scipy.stats as stats
from timing import timeit


class NaiveBayesClassifier(object):
    def __init__(self, feature_names, plotting, logging):
        self.features = feature_names
        self.num_samples = None
        self.num_classes = None
        self.mean = None
        self.variance = None
        self.priors = None
        self.plotting = plotting
        self.logging = logging

    @timeit
    def train(self, samples, labels):
        # Number of samples used to fit data
        self.num_samples = samples.size
        # List of all feature names, omitting 'class' for the labels
        self.features = samples.columns
        # Number of class values in the set of labels
        self.num_classes = len(labels['class'].unique())
        # Matrix of means, 2x11 in our case
        self.mean = np.zeros((self.num_classes, len(self.features)))
        # Matrix of variances, 2x11 in our case
        self.variance = np.zeros((self.num_classes, len(self.features)))
        # Vector of priors, length of 2 in our case for labels 0 and 1
        self.priors = np.zeros(self.num_classes)

        # Iterate over all class values
        for class_value in range(self.num_classes):
            merged = samples
            merged['class'] = labels.values
            x = samples[samples['class'] == class_value]
            x = x.drop(['class'], axis=1)

            self.mean[class_value, :] = np.mean(x.values, axis=0)
            self.variance[class_value, :] = np.var(x.values, axis=0)
            self.priors[class_value] = x.size / self.num_samples

        if self.plotting:
            for y in range(self.num_classes):
                for x in range(len(self.features)):
                    mu = self.mean[y][x]
                    var = self.variance[y][x]
                    x = np.linspace(mu - 3 * var, mu + 3 * var, 100)
                    plt.plot(x, stats.norm.pdf(x, mu, var))
                    plt.show()

    @staticmethod
    def probability(x, mu, var):
        return np.exp(-0.5 * ((x - mu) ** 2 / var)) / np.sqrt(2 * np.pi * var)

    # A function to classify a single sample by multiplying the prior, posteriors
    # for each class
    def classify_sample(self, x):
        # Empty list for the posterior of each feature
        posteriors = list()

        # Iterate over each class, getting the prior * posterior for each
        for class_value in range(self.num_classes):
            mu = self.mean[class_value]
            var = self.variance[class_value]
            prior = np.log(self.priors[class_value])

            # We use log(prob) rather than the probability itself
            posterior = np.sum(np.log(self.probability(x, mu, var)))
            posterior = prior + posterior
            posteriors.append(posterior)

        # Return the highest value as the classification
        return np.argmax(posteriors)

    @timeit
    def classify(self, samples, labels):
        correct = 0
        predictions = list()
        for idx, row in samples.iterrows():
            prediction = self.classify_sample(row)
            sys.stdout.write(str(prediction) + '\n')
            truth = labels['class'].values[idx]
            if truth == prediction:
                correct += 1
            predictions.append(self.classify_sample(row))

        percent_correct = float(100.0 * (correct / labels.size))
        if self.logging:
            print('Classified ' + str(labels.size) + ' examples with %0.3f%% accuracy' % percent_correct)

        if self.plotting:
            cm = ConfusionMatrixDisplay.from_predictions(y_true=labels, y_pred=predictions)
            cm.plot()
            plt.show()

        return percent_correct
