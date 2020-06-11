# import re
# import time
import copy
import sys
import nltk
import string
import numpy as np
import pandas as pd
from functools import reduce
from datetime import datetime

# avoiding ellipses
# pd.set_option('mode.sim_interactive', True)
# pd.set_option('expand_frame_repr', True)
# pd.set_option('display.column_space', 2)
# pd.set_option('display.max_colwidth', sys.maxsize)
pd.set_option('display.max_columns', sys.maxsize)
pd.set_option('display.max_rows', sys.maxsize)
pd.set_option('display.width', sys.maxsize)

excluded_list = """!"#$%&\()*+,:;<=>?@[\\]^`{|}~–—‐‑"""
included_list = """'"-_’/."""


# def test_tokenizing():
#     global data
#     cleaned_sentences, symbols = preprocess_translate(sentences)
#     vocabulary_sentences_test, all_excluded_vocabulary_sentences_test = clean_tokenize(sentences, combine=False)
#     data = {"Original": sentences, "Cleaned": cleaned_sentences, "Tokenized": vocabulary_sentences_test}
#     pd.DataFrame(data).to_csv("../test_org_clean.txt", "\t")


def extract_dataset(dataset_path, stopwords_path, filterBy="Created At", trainingKey="2018", testingKey="2019",
                    classes_col="Post Type"):
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


def preprocess_translate(sentences):
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
    # Fixme: Re-delete
    # del freq["-"]
    return freq


def generate_model(unique_vocabulary, data_set, classes_col, vocabulary_col, classes_freq, smoothing):
    temp_class_frequencies = []
    temp_class_probabilities = []
    temp_data_dict = {"word": list(unique_vocabulary)}

    """List of all classes"""
    classes = np.sort(np.array(list(classes_freq.keys())))

    """For every class, get the vocabulary and frequency of all all the words"""
    for cls in classes:
        temp_sentences = list(data_set[data_set[classes_col].isin([cls])][vocabulary_col])
        vocab, excluded_vocab = clean_tokenize(temp_sentences, combine=True)
        temp_class_frequencies.append(frequency_distribution(vocab))

    """Calculating probabilities with smoothing"""
    for i in range(0, classes.__len__()):
        tem_prob = dict()
        cls_freq = temp_class_frequencies[i]
        for w in unique_vocabulary:
            if w not in cls_freq:
                tem_prob[w] = smoothing / (classes_freq[classes[i]] + (smoothing * unique_vocabulary.__len__()))
                continue
            tem_prob[w] = (cls_freq[w] + smoothing) / (
                    classes_freq[classes[i]] + (smoothing * unique_vocabulary.__len__()))
        temp_class_probabilities.append(tem_prob)

    """"Creating Dataframe to save to txt file"""
    # FIXME: remove counter not needed?
    # temp_data_dict = {"counter": list(range(0, unique_vocabulary.__len__())), "all_voc": list(unique_vocabulary)}
    for i in range(0, classes.__len__()):
        temp_data_dict[classes[i] + " freq"] = []
        temp_data_dict[classes[i]] = []
        for w in unique_vocabulary:
            if w not in temp_class_probabilities[i]:
                temp_data_dict[classes[i]].append("-")
                temp_data_dict[classes[i] + " freq"].append("-")
            else:
                temp_data_dict[classes[i]].append(temp_class_probabilities[i][w])

            if w not in temp_class_frequencies[i]:
                temp_data_dict[classes[i] + " freq"].append(smoothing)
                continue
            temp_data_dict[classes[i] + " freq"].append(temp_class_frequencies[i][w])

    """Calculating class probabilites"""
    # FIXME something wrong with class freq
    total_classes = calc_total_cls_entries(classes_freq)
    classes_prob = classes_freq.copy()
    for (k, v) in classes_freq.items():
        classes_prob[k] = v / total_classes

    return temp_class_frequencies, temp_class_probabilities, classes_prob, temp_data_dict


def calc_total_cls_entries(classes_freq):
    return reduce((lambda x, y: x + y), list(classes_freq.values()))


def store_dataframe_to_file(data_dict, csv_path, text_path):
    dataframe = pd.DataFrame(data_dict)
    if csv_path is not None:
        dataframe.to_csv(csv_path)
    with open(text_path, "w") as f:
        f.write(dataframe.__repr__())


def clean_tokenize_freq_dist(train_set, vocabulary_col, combine):
    sentences = list(train_set[vocabulary_col])
    vocabulary, excluded_symbols = clean_tokenize(sentences, combine=combine)
    vocabulary_freq = frequency_distribution(vocabulary)
    unique_vocabulary = np.sort(np.array(list(vocabulary_freq.keys())))
    return unique_vocabulary, vocabulary_freq


# TEST CODE
if __name__ == "__main__":
    debug = 0

    """Constant variables"""
    dataset_path = "../data/hns_2018_2019.csv"
    stop_words_path = "../data/stopwords.txt"
    data_filter = "Created At"
    train_key = "2018"
    test_Key = "2019"
    vocabulary_col = "Title"
    classes_col = "Post Type"
    smoothing = 0.5
    csv_path = "../output/task1.csv"
    text_path = "../output/model-2018.txt"
    vocabulary_path = "../output/vocabulary.txt"
    removed_word_path = "../output/removed_word.txt"

    """Extracting data from the dataset files"""
    train_set, test_set, train_cls_freq, test_cls_freq, stopwords = extract_dataset(dataset_path,
                                                                                    stop_words_path,
                                                                                    data_filter,
                                                                                    train_key, test_Key,
                                                                                    classes_col)

    """Get all vocabulary and frequency of all the words in TRAIN dataset"""
    train_unique_vocabulary, train_vocabulary_freq = clean_tokenize_freq_dist(train_set, vocabulary_col, True)

    # FIXME testing data extraction not working
    """Get all vocabulary and frequency of all the words in TEST dataset"""
    # test_unique_vocabulary, test_vocabulary_freq = clean_tokenize_freq_dist(train_set, vocabulary_col, False)

    """Calculate conditional probabilities for each word in every class"""
    train_cls_word_freq, train_cls_word_prob, train_cls_prob, training_data = generate_model(train_unique_vocabulary,
                                                                                             train_set,
                                                                                             classes_col,
                                                                                             vocabulary_col,
                                                                                             train_cls_freq,
                                                                                             smoothing)

    """Store probabilities data frame to file"""
    store_dataframe_to_file(training_data, csv_path=csv_path, text_path=text_path)
    """Store vocabulary data frame to file"""
    store_dataframe_to_file({"vocabulary": list(train_unique_vocabulary)}, csv_path=None, text_path=vocabulary_path)
    """Store excluded data frame to file"""
    store_dataframe_to_file({"removed": [str(i).encode('utf-8') for i in excluded_list]}, csv_path=None,
                            text_path=removed_word_path)

    # TODO: Text files: vocabulary and removed words and return carriage

    # FIXME: DELETE Test func
    # test_tokenizing()
    # print("freq: ", vocabulary_freq)
    # data = {"Cleaned": vocabulary}
    # pd.DataFrame(data).sort_values(by="Cleaned", ascending=True).to_csv("../test_cleaned.txt")
    # pd.DataFrame(data, columns=["Original", "New"]).to_csv("../test.csv")
    # timer_start = time.perf_counter()
    pass
