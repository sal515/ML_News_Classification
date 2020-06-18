# -------------------------------------------------------
# Assignment 2
# FIXME: Add student id
# Written by Salman Rahman <Student id here don't forget>
# For COMP 472 Section - ABIX – Summer 2020
# --------------------------------------------------------
import random
import time
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import processing.common as common
from processing.param import param
import processing.training as train
import processing.naive_bays_classifier as classifier


def train_and_test(freq_percent):
    """Get all vocabulary and frequency of all the words in TRAIN dataset"""
    train_unique_vocabulary, removed_words = train.train_clean_tokenize_wrapper(
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

    train_excluded_vocab = list(np.concatenate([train_excluded_vocab, removed_words]))

    """Conditional probability table as dictionary to easily access the probabilities"""
    train_model_df = train.generate_model_df(trained_data)

    """Store probabilities data frame to file"""
    common.store_dataframe_to_file(
        trained_data,
        csv_path=param.train_csv_name if param.debug else None,
        output_dir=param.output_dir,
        debug_dir=param.debug_dir,
        fileName=param.train_text_name)

    """Store vocabulary data frame to file"""
    common.store_dataframe_to_file(
        {"vocabulary": list(train_unique_vocabulary)},
        csv_path=None,
        output_dir=param.output_dir,
        debug_dir=param.debug_dir,
        fileName=param.vocabulary_name)

    """Store excluded data frame to file"""
    common.store_dataframe_to_file(
        {"removed": [str(i).encode('utf-8') for i in train_excluded_vocab]},
        csv_path=None,
        output_dir=param.output_dir,
        debug_dir=param.debug_dir,
        fileName=param.removed_word_name)

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

    """Creating the testing output dataframe with all the required columns"""
    test_classification_df = common.store_dataframe_to_file(
        test_classification_dt,
        csv_path=param.result_csv_name if param.debug else None,
        output_dir=param.output_dir,
        debug_dir=param.debug_dir,
        fileName=param.result_text_name)

    return test_classification_dt, test_classification_df, train_cls_list, train_model_df


def metrics_calcualations(classification_dt, classification_df, cls_list):
    freqDist = nltk.FreqDist(classification_dt["right_wrong"])
    overall_accuracy = round((freqDist["right"] / classification_dt["Title"].__len__()) * 100, 3)

    accuracy_list = []
    precision_list = []
    recall_list = []
    # f_measure_list = []

    for cls in cls_list:
        """Metric - Precision"""
        by_classification_df = classification_df[classification_df["Classification"].isin([cls])]
        classified_as_correct = len(list(by_classification_df["Classification"]))
        correct_classification = len(
            list((by_classification_df[by_classification_df["right_wrong"].isin(["right"])])["right_wrong"]))
        try:
            precision_list.append((correct_classification / classified_as_correct) * 100)
        except ZeroDivisionError:
            precision_list.append(0)

        """Metric - Recall"""
        by_true_classification_df = classification_df[classification_df["Correct_Classification"].isin([cls])]
        number_of_items = len(list(by_true_classification_df["Correct_Classification"]))
        correct_classification = len(
            list((by_true_classification_df[by_true_classification_df["right_wrong"].isin(["right"])])[
                     "right_wrong"]))
        try:
            recall_list.append((correct_classification / number_of_items) * 100)
        except ZeroDivisionError:
            precision_list.append(0)

    """Harmonic mean/F-measure - Assuming all the weights are same for all the provided classes"""
    hm = [(2 / ((1 / r) + (1 / p))) if r != 0 and p != 0 else 0 for p, r in zip(precision_list, recall_list)]

    print_metrics(cls_list, freqDist, hm, overall_accuracy, precision_list, recall_list)

    return freqDist, overall_accuracy, precision_list, recall_list, hm


def print_metrics(cls_list, freqDist, hm, overall_accuracy, precision_list, recall_list):
    """Result output"""
    print(
        f"\n{train_type} result frequencies: {freqDist.__repr__()} \nOverall Accuracy - ((# of Right / Total) * 100) :  {overall_accuracy} %")
    print("Precision : ", list(map(lambda x: f"{x[0]} : {round(x[1], 3)} %", zip(cls_list, precision_list))))
    print("Recall : ", list(map(lambda x: f"{x[0]} : {round(x[1], 3)} %", zip(cls_list, recall_list))))
    print("Harmonic mean : ", list(map(lambda x: f"{x[0]} : {round(x[1], 3)} %", zip(cls_list, hm))))


def random_plot_color():
    r = random.random()
    g = random.random()
    b = random.random()
    return (r, g, b)


def harmic_plot(frequency_vocabLen, percentage_vocabLen, frequency_hm_values, percentage_hm_values, cls_list):
    fig, axs = plt.subplots(2)

    frequency_lines = []
    axs[0].grid(True, "both")
    axs[0].set_title("""Harmonic Mean (uniform weight) vs Words in Vocabulary (By Frequency)""")
    axs[0].set(ylabel="Harmonic Mean %")
    # axs[0].plot(param.word_freq_threshold.frequencies, param.word_freq_threshold.frequencies_result)

    frequency_vocab_len = [i[1] for i in frequency_vocabLen]
    for i, hm in enumerate(frequency_hm_values):
        frequency_lines.append(axs[0].plot(frequency_vocab_len, hm, color=random_plot_color(), label=cls_list[i]))
    axs[0].legend(loc="right")

    percentage_lines = []
    axs[1].grid(True, "both")
    axs[1].set_title("""Harmonic Mean (uniform weight) vs Words in Vocabulary (By Top percentage)""")
    axs[1].set(xlabel="# Words in Vocabulary", ylabel="Harmonic Mean %")
    # axs[1].plot(param.word_freq_threshold.percentages, param.word_freq_threshold.percentages_result)

    percentage_vocab_len = [i[1] for i in percentage_vocabLen]
    for i, hm in enumerate(percentage_hm_values):
        percentage_lines.append(axs[1].plot(percentage_vocab_len, hm, color=random_plot_color(), label=cls_list[i]))
    axs[1].legend(loc="right")

    plt.show()


"""---------Main Driver Code---------"""

frequency_hm_values = []
percentage_hm_values = []
frequency_vocab = []
percentage_vocab = []
# cls_lists = []


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
    """Handling the infrequent words filtering experiment differntly"""
    if train_type == param.experiments.infrequent_word_filtering:
        param.isInfrequentExp = True
        param.update_frequency_thresholds()

        """Lopping through different frequencies"""
        for frequency in param.word_freq_threshold.frequencies:
            param.get_paths(train_type, "frequency", frequency)
            classification_dt, classification_df, cls_list, train_model_df = train_and_test(
                (frequency, param.word_freq_threshold.frequency_str))
            frequency_vocab.append((frequency, list(train_model_df.keys()).__len__()))
            # cls_lists.append(cls_list)

            "Frequency Results metrics"
            freqDist, overall_accuracy, precision_list, recall_list, hm = metrics_calcualations(
                classification_dt, classification_df, cls_list)
            frequency_hm_values.append(hm)
            param.word_freq_threshold.frequencies_result.append(overall_accuracy)

        """Lopping through different top percentages"""
        for percentage in param.word_freq_threshold.percentages:
            param.get_paths(train_type, "percentage", percentage)
            classification_dt, classification_df, cls_list, train_model_df = train_and_test(
                (percentage, param.word_freq_threshold.percentage_str))
            # cls_lists.append(cls_list)
            percentage_vocab.append((percentage, list(train_model_df.keys()).__len__()))

            "Percentage Results metrics"
            freqDist, overall_accuracy, precision_list, recall_list, hm = metrics_calcualations(
                classification_dt, classification_df,
                cls_list)
            percentage_hm_values.append(hm)
            param.word_freq_threshold.percentages_result.append(overall_accuracy)

        continue

    """Updating result and model file paths for all other experiments"""
    param.isInfrequentExp = False
    param.get_paths(train_type, None, None)
    classification_dt, classification_df, cls_list, train_model_df = train_and_test(None)
    metrics_calcualations(classification_dt, classification_df, cls_list)

"""Determining the total time taken to complete the experiments"""
time_taken = time.perf_counter() - timer_offset
print("\nTotal time elapsed to complete the experiments ", round(time_taken, 3), "s")

"""Plotting the accurary for infrequeny words experiments"""
frequency_hm_values = np.array(frequency_hm_values).T
percentage_hm_values = np.array(percentage_hm_values).T

# if cls_lists.__len__() == 2 and cls_lists[0] == cls_lists[1]:
print("plotted harmonics")
harmic_plot(frequency_vocab, percentage_vocab, frequency_hm_values, percentage_hm_values, cls_list)
