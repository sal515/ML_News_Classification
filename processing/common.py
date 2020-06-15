import sys
import nltk
import string
import numpy as np
import pandas as pd
from functools import reduce
from datetime import datetime

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
    extracted_category = [str(datetime.strptime(str(dt), '%Y-%m-%d %H:%M:%S').year) for dt in list(data[filterBy])]
    data[data_category] = extracted_category

    """Extract all training dataset according to filter provided as parameter"""
    training_set = data[data[data_category].isin([trainingKey.lower()])]
    train_classes_freq = nltk.FreqDist(list(training_set[classes_col]))

    """Extract all testing dataset according to filter provided as parameter"""
    testing_set = data[data[data_category].isin([testingKey.lower()])]
    # test_classes_freq = nltk.FreqDist(list(testing_set[classes_col]))

    return training_set, testing_set, train_classes_freq, stop_words


def tokenize(untokenized_string):
    """Tokenize the title strings into separate words"""
    tweet_tokenizer = nltk.TweetTokenizer()
    # FIXME: Remove
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
        # for s in freq:
        #     if "-" in s:
        #         # FIXME: Remove
        #         # del s["-"]
        #         pass
        return freq

    freq = dict(nltk.FreqDist(tokenized_words))
    # Fixme: Remove
    # del freq["-"]
    return freq


def calc_total_cls_entries(classes_freq):
    """Calculating the cumulative number of classes entries in the dataset"""
    return reduce((lambda x, y: x + y), list(classes_freq.values()))


def store_dataframe_to_file(data_dict, csv_path, text_path):
    """Save the dataframe to txt file, and csv on debug"""
    dataframe = pd.DataFrame(data_dict)
    if csv_path is not None:
        dataframe.to_csv(csv_path)
    with open(text_path, "w") as f:
        f.write(dataframe.__repr__())


def clean_tokenize_wrapper(
        data_set,
        vocabulary_col,
        excluded_list,
        included_list,
        isTrain,
        trainType,
        combine,
        stopwords):
    """This function returns the (unique words in vocabulary) if isTrain==True otherwise (frequency of words in vocabulary)"""

    """List of titles/sentences in the dataset"""
    sentences = list(data_set[vocabulary_col])

    """Create list of vocabulary and excluded symbols"""
    vocabulary, excluded_symbols = clean_tokenize(
        sentences,
        excluded_list,
        included_list,
        combine=combine)

    is_vocab_list_of_lists = False

    # if trainType == "baseline":
    #     """Create list frequency of the words in the vocabulary """
    #     vocabulary_freq = frequency_distribution(vocabulary)
    #
    #     """Checking if the vocabulary_freq is list of lists"""
    #     is_vocab_list_of_lists = isinstance(vocabulary_freq, list) and isinstance(vocabulary_freq[0], dict)
    #
    #     """if it is not train, unique vocabulary doesn't have to be generated"""
    #     if not isTrain:
    #         return vocabulary_freq
    #
    #     """Create list of unique vocabulary"""
    #     if is_vocab_list_of_lists:
    #         unique_vocabulary = np.sort(
    #             np.array(list(np.concatenate(list(map(lambda d: list(d.keys()), vocabulary_freq))))))
    #         return unique_vocabulary
    #
    #     unique_vocabulary = np.sort(np.array(list(vocabulary_freq.keys())))
    #     return unique_vocabulary

    # FIXME: Remember to handle Train call of this function
    # FIXME: Test uses unique_vocab. -> uses vocabulary_freq
    # FIXME: 1. Stop words filtering - generate unique_vocab. by filtering the stop words here

    """Create list frequency of the words in the vocabulary """
    vocabulary_freq = frequency_distribution(vocabulary)

    """Checking if the vocabulary_freq is list of lists"""
    is_vocab_list_of_lists = isinstance(vocabulary_freq, list) and isinstance(vocabulary_freq[0], dict)

    if trainType == "stopwords":
        """Removing stop words from the vocabulary"""
        if is_vocab_list_of_lists:
            """Remove stopwords from the vocabulary"""
            for word in stopwords:
                for sentence_words in vocabulary_freq:
                    if word in sentence_words:
                        del sentence_words[word]
        else:
            for word in stopwords:
                if word in vocabulary_freq:
                    del vocabulary_freq[word]



    """if it is not train, unique vocabulary doesn't have to be generated"""
    if not isTrain:
        return vocabulary_freq

    """Create list of unique vocabulary"""
    if is_vocab_list_of_lists:
        unique_vocabulary = np.sort(
            np.array(list(np.concatenate(list(map(lambda d: list(d.keys()), vocabulary_freq))))))
        return unique_vocabulary

    unique_vocabulary = np.sort(np.array(list(vocabulary_freq.keys())))
    return unique_vocabulary

    # FIXME: 2. Word length filtering - generate unique_vocab. by filtering the words outside the len size
    # FIXME: 3. Infrequent word filtering - ??

    # FIXME: Remember to handle Test call of this function
    # FIXME: Test uses vocab_freq
