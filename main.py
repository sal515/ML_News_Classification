# -------------------------------------------------------
# Assignment 2
# FIXME: Add student id
# Written by Salman Rahman <Student id here don't forget>
# For COMP 472 Section - ABIX – Summer 2020
# --------------------------------------------------------
import time

import nltk
import numpy as np
import pandas as pd
import processing.training as train
import processing.common as common
import processing.naive_bays_classifier as classifier
from processing.input import param


def train_and_test(freq_percent):
    """Get all vocabulary and frequency of all the words in TRAIN dataset"""
    train_unique_vocabulary = train.train_clean_tokenize_wrapper(
        train_set,
        param.vocabulary_col,
        param.excluded_list,
        param.included_list,
        train_types=param.experiments,
        train_type=train_type,
        stopwords=stopwords,
        word_lengths=param.word_lengths,
        freq_percent=freq_percent,
        word_freq_threshold=param.word_freq_threshold)

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
    if train_type != param.experiments.infrequent_word_filtering:
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
    test_vocabulary_freq = classifier.test_clean_tokenize_wrapper(
        test_set,
        param.vocabulary_col,
        param.excluded_list,
        param.included_list)

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

    """Result output"""
    print(f"{train_type} result frequencies: ", nltk.FreqDist(test_classification_dt["right_wrong"]).__repr__())
    if train_type != param.experiments.infrequent_word_filtering:
        """Creating the testing output dataframe with all the required columns"""
        common.store_dataframe_to_file(
            test_classification_dt,
            csv_path=param.result_csv_path if param.debug else None,
            text_path=param.result_text_path)


"""---------Data Extraction---------"""

"""Get training and testing dataset from the dataset files"""
train_set, test_set, train_cls_freq, stopwords = common.extract_dataset(
    param.dataset_path,
    param.stop_words_path,
    param.data_filter,
    param.train_key,
    param.test_Key,
    param.classes_col)

"""---------Training and Testing---------"""
timer_offset = time.perf_counter()

for train_type in param.experiments.train_types:
    """Updating result and model file paths for the experiments"""
    param.get_paths(train_type)

    if train_type == param.experiments.infrequent_word_filtering:
        param.update_frequency_thresholds()
        for frequency in param.word_freq_threshold.frequencies:
            train_and_test((frequency, param.word_freq_threshold.frequency_str))
        for percentage in param.word_freq_threshold.percentages:
            train_and_test((percentage, param.word_freq_threshold.percentage_str))
        continue

    train_and_test(None)

time_taken = time.perf_counter() - timer_offset
print("\nTotal time elapsed to complete the experiments ", round(time_taken, 3), "s")
