import nltk
import processing.common
import processing.training as train
import processing.common as common
import processing.naive_bays_classifier as classifier
from processing.input import param

# debug = 0
#
# """Dataset File paths"""
# dataset_path = "./data/hns_2018_2019.csv"
# stop_words_path = "./data/stopwords.txt"
#
# """Constant variables"""
# excluded_list = """!"#$%&\()*+,:;<=>?@[\\]^`{|}~–—‐‑"""
# included_list = """'"-_’/."""
# data_filter = "Created At"
# train_key = "2018"
# test_Key = "2019"
# vocabulary_col = "Title"
# classes_col = "Post Type"
# smoothing = 0.5
# log_base = 10
#
# """Output file paths"""
# train_csv_path = "./output/task1.csv"
# train_text_path = "./output/model-2018.txt"
# vocabulary_path = "./output/vocabulary.txt"
# removed_word_path = "./output/removed_word.txt"
# result_csv_path = "./output/task2.csv"
# result_text_path = "./output/baseline_result.txt"


"""---------Training---------"""

"""Get training and testing dataset from the dataset files"""
train_set, test_set, train_cls_freq, stopwords = common.extract_dataset(
    param.dataset_path,
    param.stop_words_path,
    param.data_filter,
    param.train_key,
    param.test_Key,
    param.classes_col)

"""Get all vocabulary and frequency of all the words in TRAIN dataset"""
train_unique_vocabulary = processing.common.clean_tokenize_wrapper(
    train_set,
    param.vocabulary_col,
    param.excluded_list,
    param.included_list,
    isTrain=True,
    trainType=param.trainType,
    combine=True,
    stopwords=stopwords)

"""Get conditional probabilities for each word in every class P(w|cls)"""
train_cls_prob, trained_data, train_excluded_vocab, train_cls_list = train.generate_model(
    train_unique_vocabulary,
    train_set,
    train_cls_freq,
    param.classes_col,
    param.vocabulary_col,
    param.excluded_list,
    param.included_list,
    param.smoothing)

"""Conditional probability table as dictionary to easily access the probabilities"""
train_model_df = train.generate_model_df(trained_data)

"""Store probabilities data frame to file"""
common.store_dataframe_to_file(
    trained_data,
    csv_path=param.train_csv_path if param.debug else None,
    text_path=param.train_text_path)

"""Store vocabulary data frame to file"""
common.store_dataframe_to_file(
    {"vocabulary": list(train_unique_vocabulary)},
    csv_path=None,
    text_path=param.vocabulary_path)

"""Store excluded data frame to file"""
common.store_dataframe_to_file(
    {"removed": [str(i).encode('utf-8') for i in train_excluded_vocab]},
    csv_path=None,
    text_path=param.removed_word_path)

"""Get all vocabulary and frequency of all the words in TEST dataset"""
test_vocabulary_freq = processing.common.clean_tokenize_wrapper(
    test_set,
    param.vocabulary_col,
    param.excluded_list,
    param.included_list,
    isTrain=False,
    trainType=param.trainType,
    combine=False,
    stopwords=stopwords)

"""---------Testing---------"""

"""Score calculations"""
test_cls_scores = classifier.calculate_scores(
    test_vocabulary_freq,
    train_cls_list,
    train_cls_prob,
    train_model_df,
    param.log_base)

""" Adding columns for dataframe of test ouptut"""
test_classification_dt = classifier.classify_and_generate_result(
    test_set,
    test_vocabulary_freq,
    train_cls_list,
    test_cls_scores,
    param.debug)

"""Baseline result output"""
print(f"{param.trainType} result frequencies: ", nltk.FreqDist(test_classification_dt["right_wrong"]).__repr__())

"""Creating the testing output dataframe with all the required columns"""
common.store_dataframe_to_file(
    test_classification_dt,
    csv_path=param.result_csv_path if param.debug else None,
    text_path=param.result_text_path)
