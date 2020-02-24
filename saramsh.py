import math
import numpy
import copy
import pandas as pd
import string
import sys

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from nltk import tokenize

from saramsh_corpus import stopwords, contraction_mapping

class Saramsh:
    my_static_data = "Highly static!"

    def __init__(self, data, title):
        self.data = data
        self.title = title
        self.punctuations = '''!()[]{};:,"\<>/@#$%^&*_~+'''

    def __divide_into_sentences(self, data):
        sentences = tokenize.sent_tokenize(data)
        return sentences
    
    def __remove_punctuations(self, string):
        no_punct = ""
        for char in string:
            if char not in self.punctuations:
                no_punct = no_punct + char
        string = no_punct
        return string
    
    def __remove_punctuations_in_data(self):
        self.corpus = self.__remove_punctuations(self.corpus)
    
    def __remove_punctuations_in_title(self):
        self.title = self.__remove_punctuations(self.title)
    
    def __remove_stopwords(self, string):
        no_stopwords = ""
        for char in string:
            if char not in stopwords:
                no_stopwords = no_stopwords + char
        string = no_stopwords
        return string
    
    def __remove_stopwords_in_data(self):
        self.corpus = self.__remove_stopwords(self.corpus)
    
    def __remove_stopwords_in_title(self):
        self.title = self.__remove_stopwords(self.title)
    
    def __remove_contractions(self, string):
        contractions_removed = ""
        selected_words = contraction_mapping.keys()
        for char in string:
            if char in selected_words:
                contractions_removed = contractions_removed + \
                    contraction_mapping[char]
            else:
                contractions_removed = contractions_removed + char
        string = contractions_removed
        return string
    
    def __remove_contractions_in_data(self):
        self.corpus = self.__remove_contractions(self.corpus)

    def __remove_contractions_in_title(self):
        self.title = self.__remove_contractions(self.title)

    def __preprocess(self):
        self.corpus = self.corpus.lower()
        self.corpus = self.corpus.encode('ascii', 'ignore')
        self.corpus = self.corpus.decode("utf-8")
        
        self.__remove_punctuations_in_data()
        self.__remove_punctuations_in_title()
        self.__remove_stopwords_in_data()
        self.__remove_stopwords_in_title()
        self.__remove_contractions_in_data()
        self.__remove_contractions_in_title()

    def __get_feature_names(self, li):
        '''
        This function returns list of feature names in the given corpus
        '''
        x = []
        words = []
        for i in li:
            for j in i:
                if j != ' ' and j != ',':
                    x.append(j)
                if j == ' ' or j == ',':
                    words.append(''.join(x))
                    x = []
            words.append(''.join(x))
            x = []
        return words

    def __get_unique_features(self, li):
        '''
        This returns the unique feature names in the list
        '''
        unique = []
        for i in li:
            if i not in unique:
                unique.append(i)
        return unique

    def __get_frequency_counts(self, corpus, vector):
        '''
        This functon returns the frquency counts for each row
        '''
        f = copy.deepcopy(vector)
        x = []
        for i in corpus:
            row = i.split()
            for column in row:
                f[column] += 1
            x.append(f)
            f = copy.deepcopy(vector)
        #print("The frquency values for features in each row are :\n\n")
        # for k in x:
        # print(k)
        return x

    def __get_row_lengths(self, corpus):
        '''
        Returns the lenth of each document in corpus
        '''
        l = []
        for i in corpus:
            x = i.split()
            l.append(len(x))
        return l

    def __compute_tf(self, corpus, fc):
        '''
        Returns the Tf calculated matrix
        '''
        row_lengths = self.__get_row_lengths(corpus)
        k = 0
        for row in fc:
            for value in row:
                row[value] /= float(row_lengths[k])
            k += 1
        #print("\n\n The TF values calculated are :\n\n")
        # print(fc)
        return fc

    def __get_ni(self, corpus, vector):
        '''
        Returns the value, each feature present in how many documents
        '''
        k = 0
        count = 0
        li = list(vector)
        li_count = []
        for i in li:
            for j in corpus:
                row = j.split()
                if i in row:
                    count += 1
            li_count.append(count + 1)
            count = 0
        return li_count

    def __compute_idf(self, corpus, vector, fc, n):
        '''
        computes the idf and returns a dictionary
        '''
        k = 0
        x = []
        idf = copy.deepcopy(vector)
        ni = self.__get_ni(corpus, vector)
        for i in ni:
            x.append(numpy.log(n / ni[k]) + 1)
            k += 1
        k = 0
        for i in idf:
            idf[i] = x[k]
            k += 1
        #print("\n\n The IDF values calculated are :\n\n")
        # print(idf)
        return idf

    def __compute_tf_idf(self, tf, idf, title):
        '''
        This returns computed tf-idf dictionary for the given vocab
        '''
        x = {}
        tf_idf = []
        for key in tf:
            for value in key:
                x[value] = key[value] * idf[value]
                if value in self.title:
                    x[value] += 0.5
            tf_idf.append(x)
            x = {}
        #print("\n\n The TF * IDF values calculated are :\n\n")
        # print(tf_idf)
        return tf_idf

    def __count_words_in_each_sentence(self, sentences):
        sent_count = []
        for sent in sentences:
            sent_count.append(len(sent.split()))
        return sent_count

    def __score_sentences(self, tf, tf_idf, sent_word_count):
        sentenceValue = {}
        x = 0
        i = 0
        for sent in tf_idf:
            total_score_per_sentence = 0
            for word, score in sent.items():
                total_score_per_sentence += score
            sentenceValue[x] = total_score_per_sentence / sent_word_count[i]
            i += 1
            x += 1
        return sentenceValue

    def __find_average_score(self, sentenceValues):
        sumValues = 0
        for entry in sentenceValues:
            sumValues += sentenceValues[entry]
        # Average value of a sentence from original summary_text
        average = (sumValues / len(sentenceValues))
        print(average)
        return average

    def __generate_summary(self, sentences, sentenceValues, threshold):
        summary = ""
        for sent, score in sentenceValues.items():
            if score >= threshold:
                summary += sentences[sent]
        return summary

    def __transform(self, tf, idf, title):
        '''
        returns a normalized sparse matrix
        '''
        tf_idf = self.__compute_tf_idf(tf, idf, self.title)
        '''
        tfidf_values=[]
        x=[]
        for i in tf_idf:
            x.append(list(i.values()))
        tfidf_values.append(x)

        temp=numpy.array(tfidf_values).reshape(4,9)
        temp=normalize(temp,norm='l2')
        tfidf_sparse_matrix=csr_matrix(temp)
        print("\n\nThe output sparse matrix is :\n\n")
        print(tfidf_sparse_matrix)
        '''
        return tf_idf

    def __fit(self, corpus, n):
        '''
        calculates tf and idf values for extracted feature names
        '''
        words = self.__get_feature_names(corpus)
        unique_words = self.__get_unique_features(words)
        unique_words.sort()
        dim = []
        dim = [i * 0 for i in range(len(unique_words))]
        vector = dict(zip(unique_words, dim))
        #print("The features vector would be :\n\n",vector,"\n\n")
        fc = self.__get_frequency_counts(corpus, vector)
        tf = self.__compute_tf(corpus, fc)
        idf = self.__compute_idf(corpus, vector, fc, n + 1)
        return tf, idf

    def summarize(self):
        """
        Takes input a preprocessed corpus and returns the tf_idf,tf,idf,summary,sentence scores values
        """

        self.data = self.data.replace("\n", " ")
        original_title = self.title
        self.corpus = copy.deepcopy(self.data)
        self.__preprocess()
        sentences = self.__divide_into_sentences(self.corpus)


        
        self.corpus = sentences
        n = len(self.corpus)
        self.tf_, self.idf_ = self.__fit(self.corpus, n)
        self.tf_idf_ = self.__transform(self.tf_, self.idf_, self.title)

        temp = (self.data.encode('ascii', 'ignore')).decode("utf-8")
        sentences = self.__divide_into_sentences(temp)
        sent_word_count = self.__count_words_in_each_sentence(sentences)
        self.sentenceScores_ = self.__score_sentences(self.tf_, self.tf_idf_, sent_word_count)
        # print(idf)

        threshold = self.__find_average_score(self.sentenceScores_)
        self.summary = self.__generate_summary(sentences, self.sentenceScores_, threshold)
        print("\n ", original_title, "\n")
        print("\n" + self.summary)
