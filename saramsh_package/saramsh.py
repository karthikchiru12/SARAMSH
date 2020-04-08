import math
import numpy
import copy
import pandas as pd
import string
import sys

#Uncomment these lines if you want result as csr matrix...in __transform() method
#from scipy.sparse import csr_matrix
#from sklearn.preprocessing import normalize
from nltk import tokenize


"""
The below stopwords list has been taken from: https://gist.github.com/sebleier/554280
"""
stopwords = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
    've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
    "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
    'won', "won't", 'wouldn', "wouldn't"
]

contraction_mapping = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would",
    "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
    "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",
    "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",
    "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
    "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
    "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so as",
    "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
    "there'd've": "there would have", "there's": "there is", "here's": "here is", "they'd": "they would", "they'd've": "they would have",
    "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
    "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
    "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
    "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
    "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
    "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
    "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
    "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
    "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
    "you're": "you are", "you've": "you have"
}

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
        return average+0.3*average

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
