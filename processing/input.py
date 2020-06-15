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


    train_text_path = None
    vocabulary_path = None
    removed_word_path = None
    result_text_path = None
    train_csv_path = None
    result_csv_path = None

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
        min_freq = 1
        max_freq = 20
        min_top_percentage = 5
        max_top_percentage = 25
        steps = 5
        frequencies = None
        frequencies_result = None
        percentages = None
        percentages_result = None

    @staticmethod
    def get_paths(trainType):
        """Output file paths"""
        param.train_text_path = "./output/model-2018.txt" if trainType == "baseline" else f"./output/{trainType}-model.txt"
        param.vocabulary_path = "./output/vocabulary.txt" if trainType == "baseline" else f"./output/{trainType}-vocabulary.txt"
        param.removed_word_path = "./output/removed_word.txt" if trainType == "baseline" else f"./output/{trainType}-removed_word.txt"
        param.result_text_path = "./output/baseline_result.txt" if trainType == "baseline" else f"./output/{trainType}-result.txt"
        param.train_csv_path = "./output/task1.csv"
        param.result_csv_path = "./output/task2.csv"

    @staticmethod
    def update_frequency_thresholds():
        param.word_freq_threshold.frequencies = list(np.array(range(
            param.word_freq_threshold.min_freq,
            (param.word_freq_threshold.max_freq +
             param.word_freq_threshold.steps),
            param.word_freq_threshold.steps)) - 1)
        param.word_freq_threshold.frequencies[0] = 1

        param.word_freq_threshold.percentages = list(range(
            param.word_freq_threshold.min_top_percentage,
            (param.word_freq_threshold.max_top_percentage +
             param.word_freq_threshold.steps),
            param.word_freq_threshold.steps))
