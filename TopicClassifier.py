# Nic Fishman

import csv

from nltk.corpus import stopwords
from stemming.porter2 import stem

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier

import numpy as np
from scipy.sparse import coo_matrix, hstack
import re

class TopicClassifier:
    def __init__(self, train_list):
        '''
        Takes a list of files and creates the necessary objects to process them for training and classifacation.
        :param train_list: the list of files to load in and train on
        '''

        # load in the files in train_list, and process them for classifacation
        self.pages = self.load_strip_stem(train_list)

        # separate objects to create frequency vectors for each field
        self.count_body_vect = CountVectorizer()
        self.count_title_vect = CountVectorizer()
        self.count_tags_vect = CountVectorizer()

        # term-frequency times inverse document-frequency weighting, scale down the impact of tokens that occur very
        # frequently in a given corpus and that are hence empirically less informative than features that occur in a
        # small fraction of the training corpus
        self.tfidf_body_transformer = TfidfTransformer()
        self.tfidf_title_transformer = TfidfTransformer()
        self.tfidf_tags_transformer = TfidfTransformer()

        # crate the X matrix of the combined vectors of the three fields
        self.X = self.learn_transforms()

        # y vector of labels
        self.y = self.pages[:, 5]

        self.clf = self.learn_classifier()

    @staticmethod
    def load_strip_stem(file_list):
        '''

        :param file_list: the list of files to load in and process
        :return: a numpy array of the now processed data
        '''
        pages = []
        for file in file_list:
            with open(file, "rU") as site:
                # where each webpage is a line in the file
                for page in csv.reader(site):
                    for i in range(1, len(page)):
                        # replace all whitespace characters with spaces
                        page[i] = re.sub('\s+', ' ', page[i])
                        # for fields of interest: 6 = body, 2 = title, 3 = tags, 4 = label
                        if i in [2, 3, 4, 6]:
                            # if the word is not a stop word (a common english term that will not be meaningful)
                            # then stem the word, removing endings like 'ing' or 'ed' to increase training accuracy
                            page[i] = unicode(" ".join([stem(word) for word in page[i].split(" ")
                                                        if word not in stopwords.words('english')]), errors='ignore')
                    pages.append(page[0:7])
        return np.array(pages)

    def learn_transforms(self):
        '''
        Transform the stripped and stemmed text into useful frequencies for classification
        :return: matrix of the combined tfidf vectors of the three fields
        '''
        X_body = self.pages[:, 6]
        X_title = self.pages[:, 2]
        X_tags = self.pages[:, 3]

        # get the number each word in each string
        X_body_train_counts = self.count_body_vect.fit_transform(X_body)
        X_title_train_counts = self.count_title_vect.fit_transform(X_title)
        X_tags_train_counts = self.count_tags_vect.fit_transform(X_tags)

        # using the counts of each word, create tfidf transforms
        X_body_train_tfidf = self.tfidf_body_transformer.fit_transform(X_body_train_counts)
        X_title_train_tfidf = self.tfidf_title_transformer.fit_transform(X_title_train_counts)
        X_tags_train_tfidf = self.tfidf_tags_transformer.fit_transform(X_tags_train_counts)

        # combine the three vectors and return
        X_train = hstack([X_body_train_tfidf, X_title_train_tfidf, X_tags_train_tfidf]).tocsr()
        return X_train

    def learn_classifier(self):
        '''
        The classifier to learn and return. This is designed to facilitate easy changes to the classifier used.
        :return: the learned classifier
        '''
        return SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42).fit(self.X, self.y)

    def classify(self, classify_list):
        '''
        Process the data to be classified and then predict its class with the classifier
        :param classify_list: the list of documents to classify
        :return: the y vector of predicted labels
        '''
        pages = self.load_strip_stem(classify_list)
        X = self.transform_pages(pages)
        return self.clf.predict(X)

    def transform_pages(self, pages):
        '''
        Processes a new set of pages according to the fit on the training set. Usually the pages to be classified,
        as this data isn't fit, only processed.
        :param pages: the pages to be transformed
        :return: matrix of the combined tfidf vectors of the three fields
        '''
        X_body = pages[:, 6]
        X_title = pages[:, 2]
        X_tags = pages[:, 3]

        X_body_train_counts = self.count_body_vect.transform(X_body)
        X_title_train_counts = self.count_title_vect.transform(X_title)
        X_tags_train_counts = self.count_tags_vect.transform(X_tags)

        X_body_train_tfidf = self.tfidf_body_transformer.transform(X_body_train_counts)
        X_title_train_tfidf = self.tfidf_title_transformer.transform(X_title_train_counts)
        X_tags_train_tfidf = self.tfidf_tags_transformer.transform(X_tags_train_counts)

        return hstack([X_body_train_tfidf, X_title_train_tfidf, X_tags_train_tfidf]).tocsr()
