import math

import sys
import nltk
import string
import numpy as np
import pandas as pd
from functools import reduce
from datetime import datetime

import processing.common
import processing.training as train
import processing.common as common

excluded_list = """!"#$%&\()*+,:;<=>?@[\\]^`{|}~–—‐‑"""
included_list = """'"-_’/."""

debug = 0

"""Constant variables"""
log_base = 10
dataset_path = "./data/hns_2018_2019.csv"
stop_words_path = "./data/stopwords.txt"
data_filter = "Created At"
train_key = "2018"
test_Key = "2019"
vocabulary_col = "Title"
classes_col = "Post Type"
smoothing = 0.5
train_csv_path = "./output/task1.csv"
train_text_path = "./output/model-2018.txt"
vocabulary_path = "./output/vocabulary.txt"
removed_word_path = "./output/removed_word.txt"
result_csv_path = "./output/task2.csv"
result_text_path = "./output/baseline_result.txt"

"""Get training and testing dataset from the dataset files"""
train_set, test_set, train_cls_freq, test_cls_freq, stopwords = common.extract_dataset(
    dataset_path,
    stop_words_path,
    data_filter,
    train_key,
    test_Key,
    classes_col)

"""Get all vocabulary and frequency of all the words in TRAIN dataset"""
train_unique_vocabulary, train_vocabulary_freq = processing.common.clean_tokenize_freq_dist(
    train_set,
    vocabulary_col,
    excluded_list,
    included_list,
    True)

"""Get conditional probabilities for each word in every class P(w|cls)"""
train_cls_word_freq, train_cls_word_prob, train_cls_prob, training_data, excluded_vocab, train_cls_list, model_keys = train.generate_model(
    train_unique_vocabulary,
    train_set,
    classes_col,
    vocabulary_col,
    train_cls_freq,
    excluded_list,
    included_list,
    smoothing)

"""Store probabilities data frame to file"""
common.store_dataframe_to_file(
    training_data,
    csv_path=train_csv_path,
    text_path=train_text_path)

"""Store vocabulary data frame to file"""
common.store_dataframe_to_file(
    {"vocabulary": list(train_unique_vocabulary)},
    csv_path=None,
    text_path=vocabulary_path)

"""Store excluded data frame to file"""
common.store_dataframe_to_file(
    {"removed": [str(i).encode('utf-8') for i in excluded_vocab]},
    csv_path=None,
    text_path=removed_word_path)

# FIXME testing data extraction not working
"""Get all vocabulary and frequency of all the words in TEST dataset"""
test_unique_vocabulary, test_vocabulary_freq = processing.common.clean_tokenize_freq_dist(
    test_set,
    vocabulary_col,
    excluded_list,
    included_list,
    False)

# TODO : refactor to functions below this line
"""Conditional probability table as dictionary to easily access the probabilities"""
if "word" in training_data:
    model = pd.DataFrame(training_data, index=training_data["word"]).to_dict(orient="index")
else:
    raise Exception("Error: word column is not found in training data")

"""Score calculations"""
cls_scores = []
for words_freqs in test_vocabulary_freq:
    # cls = "ask_hn"
    temp = []
    for cls in train_cls_list:
        # FIXME: Missing the log(P_cls)
        score = math.log(train_cls_prob[cls], log_base)
        for word_freq in words_freqs.items():
            if word_freq[0] in model:
                score += word_freq[1] * math.log(model[word_freq[0]][cls], log_base)
                # score += w_f[1] *  model[w_f[0]][cls]
                # print(w_f[0], " -> ", model[w_f[0]][cls])
        temp.append(score)
        # print(w_f_d, " -> ", score)
    cls_scores.append(temp)

"""Classification by classifier : Determine class for each test sentence in the test set"""
classification_out_list = list(map(lambda p_list: train_cls_list[p_list.index(max(p_list))], cls_scores))

""" Adding columns for dataframe of test ouptut"""
if "Title" in test_set:
    classification_dt = {"Sentences": [str(i).encode('utf-8') for i in test_set["Title"]]}
else:
    raise Exception("Title column not found for testing dataset")

if debug: classification_dt.update({"Cleaned Sentences": test_vocabulary_freq})

classification_dt.update({"Classification": classification_out_list})
for i in range(0, train_cls_list.__len__()):
    classification_dt.update({str(train_cls_list[i]): [x[i] for x in cls_scores]})

if "Post Type" in test_set:
    classification_dt.update({"True Classification": list(test_set["Post Type"])})
else:
    raise Exception("Post Type column not found for testing dataset")

right_wrong_list = [("right" if x == y else "wrong") for x, y in
                    zip(classification_dt["Classification"], classification_dt["True Classification"])]

classification_dt.update({"right_wrong": right_wrong_list})
# TODO: Refactor to functions before this line

"""Baseline result output"""
print("Baseline result frequencies", nltk.FreqDist(classification_dt["right_wrong"]).__repr__())

"""Creating the testing output dataframe with all the required columns"""
common.store_dataframe_to_file(
    classification_dt,
    csv_path=result_csv_path,
    text_path=result_text_path)
