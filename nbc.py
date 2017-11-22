#!/usr/bin/env python3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
import gzip
from sys import argv

def parse(path):
    """
    returns a generator instance pointing the value in the gzip file
    :param path: string of the path of the gzip data
    """
    data = gzip.open(path, 'rb')
    for byte_line in data:
        yield eval(byte_line)  # return generator instance to save memory


def get_df(path):
    """
    takes path string and returns a dataframe to be used by the classifier.
    :param path: string of the path of the gzip data
    :return: pandas dataframe of the amazon reviews
    """
    i = 0
    df = {}
    for dict_item in parse(path):
        if i < 2000:
            df[i] = dict_item
            i += 1
        else:
            break
    # generate a DataFrame that has reviews and ratings
    desired = pd.DataFrame.from_dict(df, orient='columns').T  # transform the matrix since the data is "sideways" in the gzip
    desired = desired.drop(['asin', 'helpful', 'reviewTime', 'reviewerID', 'reviewerName', 'summary', 'unixReviewTime'], axis=1)  # strip unused data
    return desired


def use_only_sample_data(reviews, labels):
    reviews_train, reviews_test, labels_train, labels_test = train_test_split(reviews, labels, random_state=1)  # split the data into testing and training

    vectorizer = CountVectorizer()  # this is required to convert text data
    training_document_term_matrix = vectorizer.fit_transform(reviews_train)  # generate document_term_matrix for training
    testing_document_term_matrix = vectorizer.transform(reviews_test)  # generate DTM got testing

    classifier = MultinomialNB()  # instantiate classifier
    classifier.fit(training_document_term_matrix, labels_train)  # train the classifier on the test_data
    label_predictions = classifier.predict(testing_document_term_matrix)  # run the test sample through the classifier

    print('Accuracy is: ' + str(metrics.accuracy_score(labels_test, label_predictions) * 100) + "%")


if __name__ == '__main__':
    dataframe = get_df(argv[1]).sort_values(by='overall')  # load amazon reviews into a DataFrame
    dataframe['label_num'] = dataframe.overall.map({5.0:1.0, 4.0:0.0, 3.0:0.0, 2.0:0.0, 1.0:0.0})  # split between 5-star an not 5-star

    reviews = dataframe.reviewText  # get messages
    labels = dataframe.label_num  # get labels
    use_only_sample_data(reviews, labels)
