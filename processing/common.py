import sys
import nltk
import string
import numpy as np
import pandas as pd
from functools import reduce
from datetime import datetime
from processing.param import param

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
    """extract dataset from csv"""
    data = pd.read_csv(dataset_path, encoding="utf-8")
    stop_words = list(pd.read_csv(stopwords_path, header=0, names=["words"], encoding="utf-8")["words"])

    """To lower case """
    for col_name in list(data.columns):
        if data[col_name].dtype not in ['int64', 'float64']:
            try:
                data[col_name] = data[col_name].str.lower()
            except:
                print(col_name, " was not of type string")

    data_category = "data_category"

    extracted_category = None

    # s = str("1/1/2018 0:59")
    # for p in string.punctuation:
    #     s = s.replace(p, "-")
    #     print(s)

    cleaned_date_time_list = []
    for dt in list(data[filterBy]):
        for p in string.punctuation:
            dt = dt.replace(p, "-")
        cleaned_date_time_list.append(dt)

    try:
        # y m d h m s
        extracted_category = [str(datetime.strptime(str(dt), '%Y-%m-%d %H-%M-%S').year) for dt in
                              cleaned_date_time_list]
    except Exception as e:
        print("debug :", e)
        try:
            # y m d h m
            extracted_category = [str(datetime.strptime(str(dt), '%Y-%m-%d %H-%M').year) for dt in
                                  cleaned_date_time_list]
        except Exception as e:
            print("debug :", e)
            try:
                # d m y h m s
                extracted_category = [str(datetime.strptime(str(dt), '%d-%m-%Y %H-%M-%S').year) for dt in
                                      cleaned_date_time_list]
            except Exception as e:
                print("debug :", e)
                try:
                    # d m y h m
                    extracted_category = [str(datetime.strptime(str(dt), '%d-%m-%Y %H-%M').year) for dt in
                                          cleaned_date_time_list]
                except Exception as e:
                    print("debug :", e)
                    try:
                        #  m d y h m s
                        extracted_category = [str(datetime.strptime(str(dt), '%m-%d-%Y %H-%M-%S').year) for dt in
                                              cleaned_date_time_list]
                    except Exception as e:
                        print("debug :", e)
                        try:
                            #  m d y h m
                            extracted_category = [str(datetime.strptime(str(dt), '%m-%d-%Y %H-%M').year) for dt in
                                                  cleaned_date_time_list]
                        except Exception as e:
                            print("debug :", e)

    if extracted_category is None:
        quit("Data formatting error: There is an issue with data format of the dataset CSV file ")

    data[data_category] = extracted_category

    """Extract all training dataset according to filter provided as parameter"""
    training_set = data[data[data_category].isin([trainingKey.lower()])]
    train_classes_freq = nltk.FreqDist(list(training_set[classes_col]))

    """Extract all testing dataset according to filter provided as parameter"""
    testing_set = data[data[data_category].isin([testingKey.lower()])]

    return training_set, testing_set, train_classes_freq, stop_words


def tokenize(untokenized_string):
    """Tokenize the title strings into separate words"""
    tweet_tokenizer = nltk.TweetTokenizer()
    # Another tokenizer - didn't handle "'s" for words
    # return nltk.word_tokenize(s)
    return tweet_tokenizer.tokenize(untokenized_string)


def preprocess_translate(sentences, excluded_list, included_list):
    """This function cleans the letters provided in the excluded list from the titles"""
    cleaned_sentence = []
    excluded_symbols = [c for c in excluded_list]
    for sentence in sentences:
        symbols = sentence.translate(str.maketrans('', '', (string.digits + string.ascii_letters))).replace(" ", "")
        for symbol in symbols:
            if symbol not in excluded_symbols and symbol not in included_list:
                excluded_symbols.append(symbol)
    exclude = "".join(str(e) for e in excluded_symbols)

    for sentence in sentences:
        sentence = sentence.replace("’", "'").replace("–", "-").replace("—", "-").replace("‐", "-")
        sentence = sentence.translate(str.maketrans('', '', exclude))
        cleaned_sentence.append(sentence)
    return cleaned_sentence, excluded_symbols


def clean_tokenize(sentences_list, excluded_list, included_list, combine):
    tokenized = []
    """Remove punctuations from the titles"""
    cleaned_sentences, excluded_symbols = preprocess_translate(sentences_list, excluded_list, included_list)

    """Tokenize each title to a list of words after cleaning"""
    for sentence in cleaned_sentences:
        tokenized.append(tokenize(sentence))
    if combine:
        tokenized = np.concatenate(tokenized)
    return tokenized, excluded_symbols


def frequency_distribution(tokenized_words):
    if isinstance(tokenized_words, list) and isinstance(tokenized_words[0], list):
        freq = [dict(nltk.FreqDist(t)) for t in tokenized_words]
        return freq

    freq = dict(nltk.FreqDist(tokenized_words))
    return freq


def calc_total_cls_entries(classes_freq):
    """Calculating the cumulative number of classes entries in the dataset"""
    return reduce((lambda x, y: x + y), list(classes_freq.values()))


def store_dataframe_to_file(data_dict, csv_path, output_dir, debug_dir, fileName):
    """Save the dataframe to txt file, and csv on debug"""
    dataframe = pd.DataFrame(data_dict)
    output_path = "".join([output_dir if not param.isInfrequentExp else param.infrequent_exp_output_dir, fileName])

    param.createDir(output_dir)
    lines = ["  ".join(np.concatenate([[str("-")], list(dataframe.columns)])).__add__("\n")]
    for i, line in enumerate(dataframe.to_numpy()):
        lines.append("  ".join(np.concatenate([[str(i)], [str(i) for i in line]])).__add__("\n"))
    with open(output_path, "w") as f:
        for line in lines:
            f.write(line)

    if param.debug:
        debug_path = "".join([debug_dir if not param.isInfrequentExp else param.infrequent_exp_debug_dir, fileName])
        if csv_path is not None:
            dataframe.to_csv(csv_path)

        param.createDir(debug_dir)
        with open(debug_path, "w") as f:
            f.write(dataframe.__repr__())

    return dataframe
