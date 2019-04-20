# Naive Bayes Classifier and Evaluation
# Jwu-Hsuan Hwang

import os
import math
import numpy as np
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.tokenize import word_tokenize

class NaiveBayes():
    def __init__(self):
        self.class_dict = {0: 'neg', 1: 'pos'}
        self.feature_dict = {}
        self.prior = None
        self.likelihood = None

    '''
    Trains a multinomial Naive Bayes classifier on a training set.
    Specifically, fills in self.prior and self.likelihood such that:
    self.prior[class] = log(P(class))
    self.likelihood[class][feature] = log(P(feature|class))
    '''
    def train(self, train_set):
        pos_count = 0
        neg_count = 0
        pos_wordcount = 0
        neg_wordcount = 0
        pos_worddic = {}
        neg_worddic = {}
        all_worddic = {}
        # iterate over training documents
        # collect class counts and feature counts
        for root, dirs, files in os.walk(train_set):
            if root == train_set + '/pos':
                for name in files:
                    if name != '.DS_Store':
                        pos_count += 1
                        with open(os.path.join(root, name)) as f:
                            for word in f.read().split():
                                pos_wordcount += 1
                                if word not in pos_worddic:
                                    pos_worddic[word] = 1
                                else:
                                    pos_worddic[word] += 1

                                if word not in all_worddic:
                                    all_worddic[word] = 1
                                else:
                                    all_worddic[word] += 1
            else:  # neg
                for name in files:
                    if name != '.DS_Store':
                        neg_count += 1
                        with open(os.path.join(root, name)) as f:
                            for word in f.read().split():
                                neg_wordcount += 1
                                if word not in neg_worddic:
                                    neg_worddic[word] = 1
                                else:
                                    neg_worddic[word] += 1
                                if word not in all_worddic:
                                    all_worddic[word] = 1
                                else:
                                    all_worddic[word] += 1
        # features: unigram without stopwords and punctuation
        n = 0
        punctuation = ['.', ',', '?', '!', '(', ')', '"', '*', ':', ';', '&', '%', '$', '#', '@']
        for word in all_worddic:
            if word not in stopwords.words('english') and \
                    word not in punctuation and \
                    all_worddic[word] > 30:
                self.feature_dict[n] = word
                n += 1

        total_count = pos_count + neg_count

        # self.prior[class] = log(P(class))
        self.prior = np.array([math.log10(neg_count/total_count), math.log10(pos_count/total_count)])

        # add-one smoothing and take log
        # self.likelihood[class][feature] = log(P(feature|class))
        self.likelihood = np.zeros((len(self.class_dict), len(self.feature_dict)))
        for index in self.feature_dict:
            if self.feature_dict[index] in neg_worddic:
                self.likelihood[0][index] = math.log10((neg_worddic[self.feature_dict[index]] + 1) /
                                                       (neg_wordcount + len(all_worddic)))
            else:
                self.likelihood[0][index] = math.log10((0 + 1) /
                                                       (neg_wordcount + len(all_worddic)))
            if self.feature_dict[index] in pos_worddic:
                self.likelihood[1][index] = math.log10((pos_worddic[self.feature_dict[index]] + 1) /
                                                       (pos_wordcount + len(all_worddic)))
            else:
                self.likelihood[1][index] = math.log10((0 + 1) /
                                                       (pos_wordcount + len(all_worddic)))

    '''
    Tests the classifier on a development or test set.
    Returns a dictionary of filenames mapped to their correct and predicted
    classes such that:
    results[filename]['correct'] = correct class
    results[filename]['predicted'] = predicted class
    '''
    def test(self, dev_set):
        results = defaultdict(dict)
        # iterate over testing documents
        for root, dirs, files in os.walk(dev_set):
            if root == dev_set + '/pos':
                for name in files:
                    if name != '.DS_Store':
                        with open(os.path.join(root, name)) as f:
                            # create feature vectors for each document
                            feat_vec = np.zeros(len(self.feature_dict,))
                            for word in f.read().split():
                                if word in self.feature_dict.values():
                                    feat_vec[list(self.feature_dict.values()).index(word)] += 1
                            #print(feat_vec)
                            x = np.dot(self.likelihood, feat_vec)

                            results[name]['correct'] = 1
                            results[name]['predicted'] = np.argmax(x + self.prior)
            else:  # neg
                for name in files:
                    if name != '.DS_Store':
                        with open(os.path.join(root, name)) as f:
                            # create feature vectors for each document
                            feat_vec = np.zeros(len(self.feature_dict,))
                            for word in f.read().split():
                                if word in self.feature_dict.values():
                                    feat_vec[list(self.feature_dict.values()).index(word)] += 1
                            #print(feat_vec)
                            x = np.dot(self.likelihood, feat_vec)
                            results[name]['correct'] = 0
                            results[name]['predicted'] = np.argmax(x + self.prior)
        return results

    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''
    def evaluate(self, results):
        # count the results
        confusion_matrix = np.zeros((len(self.class_dict),len(self.class_dict)))
        for file in results:
            if list(results[file].values())[0] == 0 and list(results[file].values())[1] == 0:
                confusion_matrix[0][0] += 1
            if list(results[file].values())[0] == 1 and list(results[file].values())[1] == 0:
                confusion_matrix[0][1] += 1
            if list(results[file].values())[0] == 0 and list(results[file].values())[1] == 1:
                confusion_matrix[1][0] += 1
            if list(results[file].values())[0] == 1 and list(results[file].values())[1] == 1:
                confusion_matrix[1][1] += 1
        # printing out evaluation
        print('    precision | ', 'recall   |', '   F1')
        precision_neg = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
        recall_neg = confusion_matrix[0][0] /(confusion_matrix[0][0] + confusion_matrix[1][0])
        f1_neg = (2*precision_neg*recall_neg) / (precision_neg + recall_neg)
        print("neg    %.2f        %.2f       %.2f" % (precision_neg, recall_neg, f1_neg))

        precision_pos = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])
        recall_pos = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])
        f1_pos = (2 * precision_pos * recall_pos) / (precision_pos + recall_pos)
        print("pos    %.2f        %.2f       %.2f" % (precision_pos, recall_pos, f1_pos))
        accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / (confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[1][0] + confusion_matrix[1][1])
        print('accuracy: %.2f' % accuracy)

if __name__ == '__main__':
    nb = NaiveBayes()
    # make sure these point to the right directories
    nb.train('movie_reviews/train')

    results = nb.test('movie_reviews/dev')
    nb.evaluate(results)
