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

    """Output file paths"""
    train_csv_path = "./output/task1.csv"
    train_text_path = "./output/model-2018.txt"
    vocabulary_path = "./output/vocabulary.txt"
    removed_word_path = "./output/removed_word.txt"
    result_csv_path = "./output/task2.csv"
    result_text_path = "./output/baseline_result.txt"
