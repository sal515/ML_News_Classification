import nltk
import processing.common
import processing.training as train
import processing.common as common
import processing.naive_bays_classifier as classifier

excluded_list = """!"#$%&\()*+,:;<=>?@[\\]^`{|}~–—‐‑"""
included_list = """'"-_’/."""

debug = 0

"""Dataset File paths"""
dataset_path = "./data/hns_2018_2019.csv"
stop_words_path = "./data/stopwords.txt"

"""Constant variables"""
data_filter = "Created At"
train_key = "2018"
test_Key = "2019"
vocabulary_col = "Title"
classes_col = "Post Type"
smoothing = 0.5
log_base = 10

"""Output file paths"""
train_csv_path = "./output/task1.csv"
train_text_path = "./output/model-2018.txt"
vocabulary_path = "./output/vocabulary.txt"
removed_word_path = "./output/removed_word.txt"
result_csv_path = "./output/task2.csv"
result_text_path = "./output/baseline_result.txt"

"""---------Training---------"""

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
train_cls_word_freq, train_cls_word_prob, train_cls_prob, trained_data, excluded_vocab, train_cls_list, model_keys = train.generate_model(
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
    trained_data,
    csv_path=train_csv_path if debug else None,
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

"""Get all vocabulary and frequency of all the words in TEST dataset"""
test_unique_vocabulary, test_vocabulary_freq = processing.common.clean_tokenize_freq_dist(
    test_set,
    vocabulary_col,
    excluded_list,
    included_list,
    False)

"""Conditional probability table as dictionary to easily access the probabilities"""
model_df = train.generate_model_df(trained_data)

"""---------Testing---------"""

"""Score calculations"""
cls_scores = classifier.calculate_scores(
    test_vocabulary_freq,
    train_cls_list,
    train_cls_prob,
    model_df,
    log_base)

"""Classification by classifier : Determine class for each test sentence in the test set"""
classification_out_list = classifier.classify(train_cls_list, cls_scores)

""" Adding columns for dataframe of test ouptut"""
classification_dt = classifier.output_result_dict(
    test_set,
    test_vocabulary_freq,
    classification_out_list,
    train_cls_list,
    cls_scores,
    debug)

"""Baseline result output"""
print("Baseline result frequencies: ", nltk.FreqDist(classification_dt["right_wrong"]).__repr__())

"""Creating the testing output dataframe with all the required columns"""
common.store_dataframe_to_file(
    classification_dt,
    csv_path=result_csv_path if debug else None,
    text_path=result_text_path)
