import os
import numpy as np


class param:
    debug = 0

    """Dataset File paths"""
    dataset_path = "./data/hns_2018_2019.csv"
    stop_words_path = "./data/stopwords.txt"

    """Constant variables"""
    excluded_list = """!"#$%&\()*+,:;<=>?@[\\]^`{|}~–—‐‑"""
    included_list = """'"-_’/."""
    data_filter = "Created At"
    train_key = "2018"
    test_Key = "2019"
    vocabulary_col = "Title"
    classes_col = "Post Type"
    smoothing = 0.5
    log_base = 10

    isInfrequentExp = False

    """Experiment types"""

    class experiments:
        baseline = "baseline"
        stopword = "stopword"
        word_length = "wordlength"
        infrequent_word_filtering = "infrequent_word_filtering"
        train_types = [baseline, stopword, word_length,
                       infrequent_word_filtering]

    """word length experiment"""

    class word_lengths:
        min_words = 2
        max_words = 9

    """infrequent word filtering experiment"""

    class word_freq_threshold:
        frequency_str = "frequency"
        percentage_str = "percentage"
        min_freq = 0
        max_freq = 20
        min_top_percentage = 0
        max_top_percentage = 25
        steps = 5
        frequencies = None
        frequencies_result = []
        percentages = None
        percentages_result = []

    """Output paths"""
    output_dir = "./output/"
    debug_dir = "./debug/"
    infrequent_exp_output_dir = "".join([output_dir, experiments.infrequent_word_filtering, "/"])
    infrequent_exp_debug_dir = "".join([debug_dir, experiments.infrequent_word_filtering, "/"])
    train_text_name = None
    vocabulary_name = None
    removed_word_name = None
    result_text_name = None
    train_csv_name = None
    result_csv_name = None

    @staticmethod
    def get_paths(trainType, frequency_or_percentage_str, val):
        """Output file paths"""
        param.train_text_name = f"model-2018.txt" if trainType == param.experiments.baseline else f"{trainType}-model.txt"

        param.vocabulary_name = f"vocabulary.txt" if trainType == param.experiments.baseline else f"{trainType}-vocabulary.txt"

        param.removed_word_name = f"removed_word.txt" if trainType == param.experiments.baseline else f"{trainType}-removed_word.txt"

        param.result_text_name = f"baseline_result.txt" if trainType == param.experiments.baseline else f"{trainType}-result.txt"
        # param.train_csv_path = f"task1.csv"
        # param.result_csv_path = f"task2.csv"

        if frequency_or_percentage_str is not None and val is not None:
            if param.debug: param.createDir(f"{param.infrequent_exp_debug_dir}/")
            param.createDir(f"{param.infrequent_exp_output_dir}/")

            """Output file paths"""
            param.train_text_name = f"{trainType}-{frequency_or_percentage_str}-{val}-model.txt"

            param.vocabulary_name = f"{trainType}-{frequency_or_percentage_str}-{val}-vocabulary.txt"

            param.removed_word_name = f"{trainType}-{frequency_or_percentage_str}-{val}-removed_word.txt"

            param.result_text_name = f"{trainType}-{frequency_or_percentage_str}-{val}-result.txt"
            # param.train_csv_path = "task1.csv"
            # param.result_csv_path = "task2.csv"

    @staticmethod
    def update_frequency_thresholds():
        param.word_freq_threshold.frequencies = list(np.array(range(
            param.word_freq_threshold.min_freq,
            (param.word_freq_threshold.max_freq +
             (2 * param.word_freq_threshold.steps)),
            param.word_freq_threshold.steps)))
        param.word_freq_threshold.frequencies.insert(1, 1)
        # param.word_freq_threshold.frequencies[1] = 1

        param.word_freq_threshold.percentages = list(range(
            param.word_freq_threshold.min_top_percentage,
            (param.word_freq_threshold.max_top_percentage +
             param.word_freq_threshold.steps),
            param.word_freq_threshold.steps))

    @staticmethod
    def createDir(dirName):
        dir = os.path.join("", dirName)
        # dir = os.path.join("./", dirName)
        if not os.path.exists(dir):
            os.mkdir(dir)
