import numpy as np
import pandas as pd

import processing.common
import processing.training as train
import processing.common as common

excluded_list = """!"#$%&\()*+,:;<=>?@[\\]^`{|}~–—‐‑"""
included_list = """'"-_’/."""

"""Constant variables"""
dataset_path = "./data/hns_2018_2019.csv"
stop_words_path = "./data/stopwords.txt"
data_filter = "Created At"
train_key = "2018"
test_Key = "2019"
vocabulary_col = "Title"
classes_col = "Post Type"
smoothing = 0.5
csv_path = "./output/task1.csv"
text_path = "./output/model-2018.txt"
vocabulary_path = "./output/vocabulary.txt"
removed_word_path = "./output/removed_word.txt"

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
train_cls_word_freq, train_cls_word_prob, train_cls_prob, training_data, excluded_vocab, cls_list, model_keys = train.generate_model(
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
    csv_path=csv_path,
    text_path=text_path)

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

# TODO : refactor to functions
"""Conditional probability table as dictionary to easily access the probabilities"""
if "word" in training_data:
    model = pd.DataFrame(training_data, index=training_data["word"]).to_dict(orient="index")
else:
    raise Exception("Error: word column is not found in training data")

"""Score calculations"""
classification_dt = {"Sentence Vocab": test_vocabulary_freq}

for d in test_vocabulary_freq:
    cls = "ask_hn"
    # for cls in tracl
    score = 0
    for w in d.items():
        if w[0] in model:
            score += model[w[0]][cls]
    print(d, score)


# score = model[list(test_vocabulary_freq[0].items())[0][0]]["ask_hn"]

# classification_df = pd.DataFrame()