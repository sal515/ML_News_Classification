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

    train_types = ["baseline", "stopword", "wordlength"]
    minwords = 2
    maxwords = 9

    @staticmethod
    def get_paths(trainType):
        """Output file paths"""
        param.train_text_path = "./output/model-2018.txt" if trainType == "baseline" else f"./output/{trainType}-model.txt"
        param.vocabulary_path = "./output/vocabulary.txt" if trainType == "baseline" else f"./output/{trainType}-vocabulary.txt"
        param.removed_word_path = "./output/removed_word.txt" if trainType == "baseline" else f"./output/{trainType}-removed_word.txt"
        param.result_text_path = "./output/baseline_result.txt" if trainType == "baseline" else f"./output/{trainType}-result.txt"
        param.train_csv_path = "./output/task1.csv"
        param.result_csv_path = "./output/task2.csv"
