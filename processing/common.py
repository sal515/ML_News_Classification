import sys
import nltk
import string
import numpy as np
import pandas as pd
from functools import reduce
from datetime import datetime

# FIXME: Remove
# avoiding ellipses
# pd.set_option('mode.sim_interactive', True)
# pd.set_option('expand_frame_repr', True)
# pd.set_option('display.column_space', 2)
# pd.set_option('display.max_colwidth', sys.maxsize)
pd.set_option('display.max_columns', sys.maxsize)
pd.set_option('display.max_rows', sys.maxsize)
pd.set_option('display.width', sys.maxsize)


def extract_dataset(
        dataset_path,
        stopwords_path,
        filterBy,
        trainingKey,
        testingKey,
        classes_col):
    data = pd.read_csv(dataset_path, encoding="utf-8")

    for col_name in list(data.columns):
        if data[col_name].dtype not in ['int64', 'float64']:
            try:
                data[col_name] = data[col_name].str.lower()
            except:
                print(col_name, " was not of type string")

    data_category = "data_category"
    extracted_category = [str(datetime.strptime(str(dt), '%Y-%m-%d %H:%M:%S').year) for dt in list(data[filterBy])]
    data[data_category] = extracted_category

    training_set = data[data[data_category].isin([trainingKey.lower()])]
    train_classes_freq = nltk.FreqDist(list(training_set[classes_col]))
    testing_set = data[data[data_category].isin([testingKey.lower()])]
    test_classes_freq = nltk.FreqDist(list(testing_set[classes_col]))
    stop_words = pd.read_csv(stopwords_path, encoding="utf-8")

    return training_set, testing_set, train_classes_freq, test_classes_freq, stop_words


def tokenize(untokenized_string):
    tweet_tokenizer = nltk.TweetTokenizer()
    # FIXME: Remove
    # return nltk.word_tokenize(s)
    return tweet_tokenizer.tokenize(untokenized_string)


def preprocess_translate(sentences, excluded_list, included_list):
    cleaned_sentence = []
    excluded_symbols = [c for c in excluded_list]
    for symbol in sentences:
        symbols = symbol.translate(str.maketrans('', '', (string.digits + string.ascii_letters))).replace(" ", "")
        for s in symbols:
            if s not in excluded_symbols and s not in included_list:
                excluded_symbols.append(s)
    exclude = "".join(str(e) for e in excluded_symbols)

    for sentence in sentences:
        sentence = sentence.replace("’", "'").replace("–", "-").replace("—", "-").replace("‐", "-")
        sentence = sentence.translate(str.maketrans('', '', exclude))
        if sentence != "":
            cleaned_sentence.append(sentence)
    return cleaned_sentence, excluded_symbols


def clean_tokenize(sentences_list, excluded_list, included_list, combine):
    tokenized = []
    cleaned_sentences, excluded_symbols = preprocess_translate(sentences_list, excluded_list, included_list)
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
    # Fixme: Re-delete
    # del freq["-"]
    return freq


def calc_total_cls_entries(classes_freq):
    return reduce((lambda x, y: x + y), list(classes_freq.values()))


def store_dataframe_to_file(data_dict, csv_path, text_path):
    dataframe = pd.DataFrame(data_dict)
    if csv_path is not None:
        dataframe.to_csv(csv_path)
    with open(text_path, "w") as f:
        f.write(dataframe.__repr__())


def clean_tokenize_freq_dist(
        data_set,
        vocabulary_col,
        excluded_list,
        included_list,
        combine):
    sentences = list(data_set[vocabulary_col])
    vocabulary, excluded_symbols = clean_tokenize(
        sentences,
        excluded_list,
        included_list,
        combine=combine)
    vocabulary_freq = frequency_distribution(vocabulary)

    if isinstance(vocabulary_freq, list) and isinstance(vocabulary_freq[0], dict):
        unique_vocabulary = np.sort(
            np.array(list(np.concatenate(list(map(lambda d: list(d.keys()), vocabulary_freq))))))
        return unique_vocabulary, vocabulary_freq

    unique_vocabulary = np.sort(np.array(list(vocabulary_freq.keys())))
    return unique_vocabulary, vocabulary_freq