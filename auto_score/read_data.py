# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 09:32:12 2019

@author: gzh
"""
import numpy as np
import nltk
from nltk import ngrams
from textblob import Word
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textstat.textstat import textstat
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class essay_examples(object):
    def __init__(self,
                 essay_id,
                 essay_set,
                 essay,
                 essay_score):
        self.essay_id = essay_id
        self.essay_set = essay_set
        self.essay = essay
        self.essay_score = essay_score
        
    def __repr__(self):
        s = ""
        s += "essay_id: %s"%(self.essay_id)
        s += ", essay_set: %s"%(self.essay_set)
        s += ", essay: %s"%(self.essay)
        s += ", essay_score: %s"%(self.essay_score)
        return s
    
    def __str__(self):
        return self.__repr__()
    
#class features(object):
#    def __init__(self,
#                 ):
        
def read_sat_words():
    f = open('SAT_words')
    sat_word_dict = set()
    while 1:
        line = f.readline()
        if line == '':
            break
        linelist = line.strip().split()
        sat_word_dict.add(linelist[0])
#        print(sat_word_dict)
    return sat_word_dict
    
def read_examples(input_file, set_id, is_training):
    f = open(input_file, errors='replace')
    examples = []
    
    for i in range(6):
        line = f.readline()
        linelist = line.strip().split('\t')
        essay_id = linelist[0]
        essay_set = linelist[1]
        essay = linelist[2]
        essay_score = None
        if str(essay_set) == str(set_id):
            if essay[0] == '"':
                essay = essay[1:len(essay)]
            if essay[-1] == '"':
                essay = essay[0:len(essay)-1]
            if is_training:
                essay_score = linelist[6]
            
            example = essay_examples(essay_id, essay_set, essay, essay_score)
            examples.append(example)
    f.close()
    return examples

def feature_word(essay):
    sat_word_dict = read_sat_words()
    stop_words = set(nltk.corpus.stopwords.words("english"))
    word_list = nltk.word_tokenize(essay)
    stop_word_count = len([[word]for word in word_list if word in stop_words])
    sat_word_count = len([[word]for word in word_list if word in sat_word_dict])
    spell_error_count = sum([Word(word).spellcheck()[0][0] != word for word in word_list])
    long_word_count = sum([len(word) > 7 for word in word_list])
    exc_count = word_list.count('!')
    que_count = word_list.count('?')
    comma_count = word_list.count(',')
    return stop_word_count, sat_word_count, spell_error_count, long_word_count, exc_count, que_count, comma_count
            

def feature_length(essay):
    char_count = len(essay)
    word_list = essay.split(' ')
    word_count = len(word_list)
    token_for_sent = ['.', '!', '?']
    sentence_count = 0
    for tok in essay:
        if tok in token_for_sent:
            sentence_count += 1  
    average_sentence_length = 0  
    if sentence_count != 0:
        average_sentence_length = word_count / sentence_count
    return char_count, word_count, sentence_count, average_sentence_length
    

def feature_ngram(essay):
    word_list = nltk.word_tokenize(essay.strip())
    
    unigram = [gram for gram in ngrams(word_list, 1)]
    bigram = [gram for gram in ngrams(word_list, 2)]
    trigram = [gram for gram in ngrams(word_list, 3)]
    unigram_count = len([[item]for item in sorted(set(unigram))])
    bigram_count = len([[item]for item in sorted(set(bigram))])
    trigram_count = len([[item] for item in sorted(set(trigram))])
    return unigram_count, bigram_count, trigram_count
    
def feature_pos(essay):
    ##fw外来词
    noun_count, adj_count, adv_count, verb_count, fw_count = 0, 0, 0, 0, 0
    word_list = nltk.word_tokenize(essay)
    tag_list = nltk.pos_tag(word_list)
    for tag in tag_list:
        if tag[1].startswith('NN'):
            noun_count += 1
        if tag[1].startswith('JJ'):
            adj_count += 1
        if tag[1].startswith('RB'):
            adv_count += 1
        if tag[1].startswith('FW'):
            fw_count += 1
        if tag[1].startswith('VB'):
            verb_count += 1
    return noun_count, adj_count, adv_count, verb_count, fw_count

def feature_sentiment(essay):
    sentences = nltk.tokenize.sent_tokenize(essay)
    sid = SentimentIntensityAnalyzer()
    neg_sentiment, neu_sentiment, pos_sentiment = 0, 0, 0
    for sent in sentences:
        ss = sid.polarity_scores(sent)
        for k in ss.keys():
            if k == 'neg':
                neg_sentiment += ss[k]
            elif k == 'neu':
                neu_sentiment += ss[k]
            elif k == 'pos':
                pos_sentiment += ss[k]
    return neg_sentiment, neu_sentiment, pos_sentiment

def feature_readability(essay):
    syllable_count = textstat.syllable_count(essay)
    #音节数统计
    flesch_reading_ease = textstat.flesch_reading_ease(essay)
    #文档的易读性0-100之间的分数
    smog_index = textstat.smog_index(essay)
    #烟雾指数，反映文档的易读程度，更精确，更容易计算
    flesch_kincaid_index = textstat.flesch_kincaid_grade(essay)
    #等级分数，年级等级
    coleman_liau_index = textstat.coleman_liau_index(essay)
    #返回文本的年级级别
    automated_readability_index = textstat.automated_readability_index(essay)
    #自动可读性指数，接近理解文本需要的年级
    dale_chall_readability_score = textstat.dale_chall_readability_score(essay)
    #返回年级级别，使用最常见的英文单词
    difficult_words = textstat.difficult_words(essay)
    
    linsear_write_formula = textstat.linsear_write_formula(essay)
    #返回文本的年级级别
    gunning_fog = textstat.gunning_fog(essay)
    #迷雾指数， 反映文本的阅读难度
    return syllable_count, flesch_reading_ease, smog_index, flesch_kincaid_index, coleman_liau_index, automated_readability_index, dale_chall_readability_score, difficult_words, linsear_write_formula, gunning_fog
    
def model_random_forest(xtrain, xdev, ytrain, xtest, i):
    x_train = xtrain
    y_train = ytrain
    model = RandomForestRegressor(n_estimators=100, n_jobs = -1, random_state=i, oob_score=True)
    model.fit(x_train, y_train)
    y_dev = model.predict(xdev)
    y_pred = model.predict(xtest)
    return y_dev, y_pred

def model_gradient_boosting(xtrain, xdev, ytrain, xtest, i):
    x_train = xtrain
    y_train = ytrain
    model = GradientBoostingRegressor(n_estimators=100, random_state=i)
    model.fit(x_train, y_train)
    y_dev = model.predict(xdev)
    y_pred = model.predict(xtest)
    return y_dev, y_pred

def model_xg(xtrain, xdev, ytrain, xtest, i):
    xgbr = XGBRegressor(nthread=4, random_state=i)
    x_train = xtrain
    y_train = ytrain
    xgbr.fit(x_train, y_train)
    model = xgbr
    y_dev = model.predict(xdev)
    y_pred = model.predict(xtest)
    return y_dev, y_pred

def model_cv(xtrain, xdev, ytrain, xtest, i):
    x_train = xtrain
    y_train = ytrain
    model = LassoCV(random_state=i)
    model.fit(x_train, y_train)
    y_dev = model.predict(xdev)
    y_pred = model.predict(xtest)
    return y_dev, y_pred

def examples_to_features(examples):
    essay_id_list = []
    essay_scores = []
    input_example_length = len(examples)
    input_features = np.zeros((input_example_length, 31))
    for index, example in enumerate(examples):
        essay_id = example.essay_id
        essay_score = example.essay_score
        essay_scores.append(essay_score)
        essay_id_list.append(essay_id)
        essay = example.essay
        char_count, word_count, sentence_count, average_sentence_length = feature_length(essay)
#    print(char_count, word_count, sentence_count, average_sentence_length)
        stop_word_count, sat_word_count, spell_error_count, long_word_count, exc_count, que_count, comma_count = feature_word(essay)
#    print(stop_word_count, sat_word_count, spell_error_count, long_word_count, exc_count, que_count, comma_count)
        unigram_count, bigram_count, trigram_count = feature_ngram(essay)
#    print(unigram_count, bigram_count, trigram_count)
        noun_count, adj_count, adv_count, verb_count, fw_count = feature_pos(essay)
#    print( noun_count, adj_count, adv_count, verb_count, fw_count)
        neg_sentiment, neu_sentiment, pos_sentiment  =feature_sentiment(essay)
#    print(neg_sentiment, neu_sentiment, pos_sentiment)
        syllable_count, flesch_reading_ease, smog_index, flesch_kincaid_index, coleman_liau_index, automated_readability_index, dale_chall_readability_score,\
        difficult_words, linsear_write_formula, gunning_fog=feature_readability(essay)
#    print( syllable_count, flesch_reading_ease, smog_index, flesch_kincaid_index, coleman_liau_index, automated_readability_index, dale_chall_readability_score, difficult_words, linsear_write_formula, gunning_fog)
        input_features[index] = [char_count, word_count, sentence_count, average_sentence_length,
                  stop_word_count, sat_word_count, spell_error_count, long_word_count,
                  exc_count, que_count, comma_count, unigram_count, bigram_count, trigram_count,
                  noun_count, adj_count, adv_count, verb_count, fw_count, neg_sentiment, pos_sentiment,
                  syllable_count, flesch_reading_ease, smog_index, flesch_kincaid_index, coleman_liau_index, automated_readability_index, dale_chall_readability_score,
                  difficult_words, linsear_write_formula, gunning_fog]
    assert len(essay_id_list) == input_features.shape[0]
    df = pd.DataFrame(input_features, index = essay_id_list, dtype=np.float64)
    return df, essay_scores

if __name__ == '__main__':
    
    examples_train = read_examples('train.tsv', set_id=1, is_training=True)
    examples_dev = read_examples('dev.tsv', set_id=1, is_training=True)
    examples_test = read_examples('test.tsv', set_id=1, is_training=False)
    
    df_train, train_scores= examples_to_features(examples_train)    
#    print(df_train, train_scores)
    df_dev, dev_scores = examples_to_features(examples_dev)
    df_test, _ = examples_to_features(examples_test)
    
    ss = StandardScaler()
    df_train = ss.fit_transform(df_train)
    df_dev = ss.fit_transform(df_dev)
    df_test = ss.transform(df_test)
    
    df_train = df_train.astype(np.float64)
    df_dev = df_dev.astype(np.float64)
    df_test = df_test.astype(np.float64)
    
    train_scores = [float(i)for i in train_scores]
    dev_scores = [float(i)for i in dev_scores]
    
    y_dev = np.zeros([len(df_dev)])
    for i in range(3):
        xtrain, xcross, ytrain, ycross = train_test_split(df_train, train_scores, test_size = 0.2, random_state = i)
        y1_cross, y1_dev = model_xg(xtrain, xcross, ytrain, df_dev, i)
        y2_cross, y2_dev = model_cv(xtrain, xcross, ytrain, df_dev, i)
        y3_cross, y3_dev = model_gradient_boosting(xtrain, xcross, ytrain, df_dev, i)
        y4_cross, y4_dev = model_random_forest(xtrain, xcross, ytrain, df_dev, i)
        y_cross = (y1_cross + y2_cross + y3_cross + y4_cross) / 4
        y_dev += (y1_dev + y2_dev + y3_dev + y4_dev) / 4
        print(ycross, y_cross)
    y_dev = y_dev / 4
    print(dev_scores, y_dev)
    
#    print(df_train.dtype, df_dev.dtype, df_test.dtype)
#    print(train_scores, dev_scores)
    
#    y_dev, y_pred = model_xg(df_train, df_dev, train_scores, df_test, 10)
#    print(dev_scores, y_dev)
#    print(y_pred)
    

    
    
