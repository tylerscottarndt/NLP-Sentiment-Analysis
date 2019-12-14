import numpy as np
import sys
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split

stop_words = set(stopwords.words('english'))
np.set_printoptions(threshold=sys.maxsize)


class DataFormatter:
    def save_files_as_numpy(self, pos_text_file, neg_text_file):
        print("Loading data...")
        pos_reviews = open(pos_text_file).read().split('\n')
        neg_reviews = open(neg_text_file).read().split('\n')

        # clean the data
        pos_reviews = self.__clean_data(pos_reviews)
        neg_reviews = self.__clean_data(neg_reviews)

        pos_labels = [1]*len(pos_reviews)
        neg_labels = [-1]*len(neg_reviews)

        labels = np.asarray(pos_labels + neg_labels)
        reviews = np.asarray(pos_reviews + neg_reviews)

        # split data into 80% train and 20% test
        print("Splitting data...")
        x_train, x_test, y_train, y_test = train_test_split(
            reviews, labels, test_size=0.2, random_state=1, stratify=labels)

        # save numpy arrays
        print("Saving...")
        np.savez('review_data.npz',
                 x_train=x_train,
                 y_train=y_train,
                 x_test=x_test,
                 y_test=y_test)

        print("Saved!")

    def __clean_data(self, arr):
        print("Cleaning data...")
        port_stemmer = PorterStemmer()
        result = []
        for line in arr:
            words = line.split(" ")
            words = [port_stemmer.stem(token) for token in words if token not in stop_words]
            words = " ".join(words)
            result.append(words)

        return result

    def vectorize(self, samples, features):
        print("Vectorizing data...")
        feature_vector = []
        for line in samples:
            vector = np.zeros(len(features), dtype=int)
            words = line.split()
            for word in words:
                if word in features:
                    vector[features.index(word)] += 1
            feature_vector.append(vector)

        return feature_vector
