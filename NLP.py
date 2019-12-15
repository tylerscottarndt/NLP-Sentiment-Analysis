import numpy as np
import matplotlib.pyplot as plt
from DataFormatter import DataFormatter
from sklearn.svm import SVC
data_formatter = DataFormatter()
data_files = np.load('review_data.npz')
param_files = np.load('optimal_parameters.npz')

class NLP:
    def __init__(self):
        self.x_train = data_files['x_train']
        self.y_train = data_files['y_train']
        self.x_test = data_files['x_test']
        self.y_test = data_files['y_test']
        self.vocabulary = self.__generate_vocabulary(self.x_train)
        self.vocab_lower_thresh = param_files['optimal_lower_thresh']
        self.vocab_upper_thresh = param_files['optimal_upper_thresh']
        self.c_val = param_files['optimal_c_val']

    def __generate_vocabulary(self, x_train):
        dict = {}
        for line in x_train:
            words = line.split()
            unique_words = set(words)
            for uw in unique_words:
                if uw not in dict:
                    dict[uw] = 1
                else:
                    dict[uw] += 1

        return dict

    def generate_bag_of_words(self, dict, lower_thresh, upper_thresh):
        print("Generating bag-of-words...")
        bag_of_words = []
        for k, v in dict.items():
            if lower_thresh < v < upper_thresh:
                bag_of_words.append(k)

        return bag_of_words

    def train_on_grid_search(self, thresh_val_range, c_val_range):
        optimal_thresh_vals = []
        optimal_c_val = ""
        highest_accuracy = 0
        accuracy_results = []
        for thresh_val in thresh_val_range:
            bag_of_words = self.generate_bag_of_words(self.vocabulary, thresh_val[0], thresh_val[1])
            training_vector = data_formatter.vectorize(self.x_train, bag_of_words)
            for c_val in c_val_range:
                accuracy = self.__tune_on_kfolds(5, training_vector, nlp.y_train, c_val)
                print("ACCURACY: %" + str(accuracy))
                print("Lower-Threshold: " + str(thresh_val[0]))
                print("Upper-Threshold: " + str(thresh_val[1]))
                print("C-Value: " + str(c_val) + "\n")

                if accuracy > highest_accuracy:
                    optimal_thresh_vals = thresh_val
                    optimal_c_val = c_val
                    highest_accuracy = accuracy

        print("WINNER: %" + str(highest_accuracy))
        print("Lower-Threshold: " + str(optimal_thresh_vals[0]))
        print("Upper-Threshold: " + str(optimal_thresh_vals[1]))
        print("C-Value: " + str(optimal_c_val))

        plt.plot(accuracy_results, marker=".")
        plt.title("Correct Predictions over Grid Search", fontsize=15)
        plt.xlabel("Tests")
        plt.ylabel("Correct Predictions")
        plt.show()

        print("Saving...")
        np.savez('optimal_parameters.npz',
                 optimal_lower_thresh=optimal_thresh_vals[0],
                 optimal_upper_thresh=optimal_thresh_vals[1],
                 optimal_c_val=optimal_c_val)

        print("Saved!")

    def __tune_on_kfolds(self, k_folds, sample_vector, targets, c_val):
        print("Tuning on K-Folds...")
        svm = SVC(kernel='linear', random_state=1)
        svm.C = c_val
        correct_predictions = 0

        # split samples and targets into k-folds
        samples = np.array_split(sample_vector, k_folds)
        targets = np.array_split(targets, k_folds)

        # iterate over the k-folds
        for i in range(k_folds):
            print("Fold " + str(i+1))
            test_fold_data = samples[i]
            training_folds_data = samples.copy()
            del training_folds_data[i]
            training_folds_data = np.concatenate(training_folds_data, axis=0)

            test_fold_targets = targets[i]
            training_folds_targets = targets.copy()
            del training_folds_targets[i]
            training_folds_targets = np.concatenate(training_folds_targets, axis=0)

            svm.fit(training_folds_data, training_folds_targets)
            predictions = svm.predict(test_fold_data)

            for p, _ in enumerate(predictions):
                if predictions[p] == test_fold_targets[p]:
                    correct_predictions += 1

        '''return accuracy value'''
        return correct_predictions/len(sample_vector)


if __name__ == '__main__':
    # data_formatter.save_files_as_numpy("rt-polarity.pos.txt", "rt-polarity.neg.txt")
    nlp = NLP()

    thresh_val_range = [[5, 2000], [25, 1000], [50, 500]]
    c_val_range = [0.0001, 1, 1000]
    # nlp.train_on_grid_search(thresh_val_range, c_val_range)

    '''predicting results on test set'''
    bag_of_words = nlp.generate_bag_of_words(nlp.vocabulary, nlp.vocab_lower_thresh, nlp.vocab_upper_thresh)
    training_vector = data_formatter.vectorize(nlp.x_train, bag_of_words)
    testing_vector = data_formatter.vectorize(nlp.x_test, bag_of_words)
    svm = SVC(kernel='linear', random_state=1)
    svm.C = nlp.c_val
    print("Training svm...")
    svm.fit(training_vector, nlp.y_train)
    print("Predicting model...")
    test_predictions = svm.predict(testing_vector)
    correct_predictions = 0
    for i, _ in enumerate(test_predictions):
        if test_predictions[i] == nlp.y_test[i]:
            correct_predictions += 1

    print("Final Accuracy: %" + str(correct_predictions / len(test_predictions) * 100))
