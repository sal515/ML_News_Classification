# import re
# import time
import copy
import sys
import nltk
import string
import numpy as np
import pandas as pd
from datetime import datetime

# avoiding ellipses
pd.set_option('display.max_colwidth', sys.maxsize)
pd.set_option('display.max_rows', sys.maxsize)

excluded_list = """!"#$%&\()*+,:;<=>?@[\\]^`{|}~–—‐‑"""
included_list = """'"-_’/."""


def test_tokenizing():
    global data
    cleaned_sentences, symbols = preprocess_translate(sentences)
    vocabulary_sentences_test, all_excluded_vocabulary_sentences_test = clean_tokenize(sentences, combine=False)
    data = {"Original": sentences, "Cleaned": cleaned_sentences, "Tokenized": vocabulary_sentences_test}
    pd.DataFrame(data).to_csv("../test_org_clean.txt", "\t")


def extract_dataset(dataset_path, stopwords_path, filterBy="Created At", trainingKey="2018", testingKey="2019",
                    classes_col="Post Type"):
    data = pd.read_csv(dataset_path, encoding="utf-8")

    for col_name in list(data.columns):
        if data[col_name].dtype not in ['int64', 'float64']:
            try:
                data[col_name] = data[col_name].str.lower()
            except:
                print(col_name, " was not of type string")

    data_categ = "data_cat"
    extracted_category = [str(datetime.strptime(str(dt), '%Y-%m-%dt %H:%M:%S').year) for dt in list(data[filterBy])]
    data[data_categ] = extracted_category

    training_set = data[data[data_categ].isin([trainingKey.lower()])]
    testing_set = data[data[data_categ].isin([testingKey.lower()])]
    classes_freq = nltk.FreqDist(list(data[classes_col]))
    stop_words = pd.read_csv(stopwords_path, encoding="utf-8")

    return training_set, testing_set, classes_freq, stop_words


def tokenize(untokenized_string):
    tweet_tokenizer = nltk.TweetTokenizer()
    # FIXME: Remove
    # return nltk.word_tokenize(s)
    return tweet_tokenizer.tokenize(untokenized_string)


def preprocess_translate(sentences):
    cleaned_sentence = []
    excluded_symbols = [c for c in excluded_list]
    for symbol in sentences:
        symbols = symbol.translate(str.maketrans('', '', (string.digits + string.ascii_letters))).replace(" ", "")
        for s in symbols:
            if s not in excluded_symbols and s not in included_list:
                excluded_symbols.append(s)
    exclude = "".join(str(e) for e in excluded_symbols)

    # FIXME: combined replace
    for sentence in sentences:
        sentence = sentence.replace("’", "'").replace("–", "-").replace("—", "-").replace("‐", "-")
        sentence = sentence.translate(str.maketrans('', '', exclude))
        if sentence != "":
            cleaned_sentence.append(sentence)
    return cleaned_sentence, excluded_symbols


def clean_tokenize(sentences_list, combine):
    tokenized = []
    cleaned_sentences, excluded_symbols = preprocess_translate(sentences_list)
    for s in cleaned_sentences:
        tokenized.append(tokenize(s))

    if combine:
        tokenized = np.concatenate(tokenized)
    return tokenized, excluded_symbols


def frequency_distribution(tokenized_words):
    if isinstance(tokenized_words, list) and isinstance(tokenized_words[0], list):
        freq = [dict(nltk.FreqDist(t)) for t in tokenized_words]
        for s in freq:
            if "-" in s:
                # FIXME: "Add ' "
                # FIXME: "Add _ "
                # del s["-"]
                pass
        return freq

    freq = dict(nltk.FreqDist(tokenized_words))
    # Fixme Re delete
    # del freq["-"]
    return freq


# TEST CODE
if __name__ == "__main__":
    debug = 0

    """Constant variables"""
    dataset_path = "../data/hns_2018_2019.csv"
    stopword_path = "../data/stopwords.txt"
    data_filter = "Created At"
    train_key = "2018"
    test_Key = "2019"
    vocabulary_col = "Title"
    classes_col = "Post Type"

    """Extracting data from the dataset files"""
    train_set, test_set, classes_dict, stopwords = extract_dataset(dataset_path, stopword_path, data_filter, train_key,
                                                                   test_Key, classes_col)

    """Get all vocabulary and frequency of all the words in dataset"""
    sentences = list(train_set[vocabulary_col])
    vocabulary, all_excluded_vocabulary = clean_tokenize(sentences, combine=True)
    vocabulary_freq = frequency_distribution(vocabulary)

    """For every class, get the vocabulary and frequency of all all the words"""
    classes = list(classes_dict.keys())
    # FIXME : Remove
    # classes_vocab = []
    class_frequencies = []
    class_probabilities = []

    for cls in classes:
        temp_sentences = list(train_set[train_set[classes_col].isin([cls])][vocabulary_col])
        # FIXME : Not handling  excluded vocubulary yet
        vocab, excluded_vocab = clean_tokenize(temp_sentences, combine=True)
        # FIXME : Remove
        # classes_vocab.append(voc)
        class_frequencies.append(frequency_distribution(vocab))

    for i in range(0, classes.__len__()):
        tem_probs = dict()
        for (k, v) in class_frequencies[i].items():
            tem_probs[k] = v / classes_dict[classes[i]]
        class_probabilities.append(tem_probs)

    """"Creating Dataframe to save to txt file"""
    all_voc = np.sort(np.array(list(vocabulary_freq.keys())))

    training_data = {"counter": list(range(0, all_voc.__len__())), "all_voc": list(all_voc)}
    for i in range(0, classes.__len__()):
        training_data[classes[i]] = []
        for w in list(all_voc):
            if w not in class_probabilities[i]:
                training_data[classes[i]].append("-")
                continue
            training_data[classes[i]].append(class_probabilities[i][w])

    """Store dataframes to files"""
    training_dataframe = pd.DataFrame(training_data)
    training_dataframe.to_csv("../task1.txt", "\t")
    with open("../model-2018.txt", "w") as f:
        f.write(training_dataframe.__repr__())

    # FIXME: DELETE Test func
    # test_tokenizing()

    # FIXME
    # print("freq: ", vocabulary_freq)

    # FIXME
    # data = {"Cleaned": vocabulary}
    # pd.DataFrame(data).sort_values(by="Cleaned", ascending=True).to_csv("../test_cleaned.txt")

    # pd.DataFrame(data, columns=["Original", "New"]).to_csv("../test.csv")

    # Fixme
    # timer_start = time.perf_counter()

    pass
