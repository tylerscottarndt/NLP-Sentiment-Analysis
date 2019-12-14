import numpy as np
import matplotlib.pyplot as plt
from DataFormatter import DataFormatter
from sklearn.svm import SVC
data_formatter = DataFormatter()
data_files = np.load('review_data.npz')


class NLP:
    def __init__(self):
        self.x_train = data_files['x_train']
        self.y_train = data_files['y_train']
        self.x_test = data_files['x_test']
        self.y_test = data_files['y_test']
        self.vocabulary = self.__generate_vocabulary(self.x_train)

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

    def tune_on_kfolds(self, k_folds, sample_vector, targets, c_val):
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

        # return accuracy value
        return correct_predictions/len(sample_vector)


if __name__ == '__main__':
    # data_formatter.save_files_as_numpy("rt-polarity.pos.txt", "rt-polarity.neg.txt")
    nlp = NLP()

    thresh_val_range = [[5, 2000], [25, 1000], [50, 500]]
    c_val_range = [0.0001, 1, 1000]

    best_thresh_val = ""
    best_c_val = ""
    accuracy_results = []
    highest_accuracy = 0
    for thresh_val in thresh_val_range:
        bag_of_words = nlp.generate_bag_of_words(nlp.vocabulary, thresh_val[0], thresh_val[1])
        training_vector = data_formatter.vectorize(nlp.x_train, bag_of_words)
        testing_vector = data_formatter.vectorize(nlp.x_test, bag_of_words)
        accuracy = ""
        for c_val in c_val_range:
            accuracy = nlp.tune_on_kfolds(5, training_vector, nlp.y_train, c_val)
            accuracy_results.append(accuracy)
            print("ACCURACY: %" + str(accuracy))
            print("Lower-Threshold: " + str(thresh_val[0]))
            print("Upper-Threshold: " + str(thresh_val[1]))
            print("C-Value: " + str(c_val) + "\n")

            if accuracy > highest_accuracy:
                best_thresh_val = thresh_val
                best_c_val = c_val
                highest_accuracy = accuracy

    print("WINNER: %" + str(highest_accuracy))
    print("Lower-Threshold: " + str(best_thresh_val[0]))
    print("Upper-Threshold: " + str(best_thresh_val[1]))
    print("C-Value: " + str(best_c_val))

    plt.plot(accuracy_results, marker=".")
    plt.title("Correct Predictions over Grid Search", fontsize=15)
    plt.xlabel("Tests")
    plt.ylabel("Correct Predictions")
    plt.show()

    # predicting results on test set
    bag_of_words = nlp.generate_bag_of_words(nlp.vocabulary, best_thresh_val[0], best_thresh_val[1])
    training_vector = data_formatter.vectorize(nlp.x_train, bag_of_words)
    testing_vector = data_formatter.vectorize(nlp.x_test, bag_of_words)
    svm = SVC(kernel='linear', random_state=1)
    svm.C = best_c_val
    print("Training model on ...")
    svm.fit(training_vector, nlp.y_train)
    print("Predicting model...")
    test_predictions = svm.predict(testing_vector)
    correct_predictions = 0
    for i,_ in enumerate(test_predictions):
        if test_predictions[i] == nlp.y_test[i]:
            correct_predictions += 1

    print("Final Accuracy: %" + str(correct_predictions / len(test_predictions) * 100))
