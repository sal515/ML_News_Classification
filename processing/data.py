# import re
# import time
import nltk
import numpy as np
import pandas as pd


# import string


# def preprocess_regex(sentences):
#     clean_sentence = []
#     # for sentence in (sentences[:][0]):  # this is Df_pd for Df_np (text[:])
#     for sentence in sentences:  # this is Df_pd for Df_np (text[:])
#         new_sentence = re.sub("<.*?>", "", sentence)  # remove HTML tags
#         new_sentence = re.sub("[^\w\s]", "", new_sentence)  # remove punc.
#         new_sentence = re.sub("\d+", "", new_sentence)  # remove numbers
#         # new_sentence = new_sentence.lower()  # lower case, .upper() for upper
#         if new_sentence != "":
#             clean_sentence.append(new_sentence)
#     return clean_sentence


def preprocess_translate(sentences):
    clean_sentence = []
    # excluded_punc = string.punctuation
    # for sentence in (sentences[:][0]):  # this is Df_pd for Df_np (text[:])
    for sentence in sentences:  # this is Df_pd for Df_np (text[:])
        new_sentence = sentence.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'))
        new_sentence = new_sentence.translate(str.maketrans('', '', '0123456789'))
        # new_sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        # new_sentence = new_sentence.translate(str.maketrans('', '', string.digits))
        # new_sentence = re.sub(" +", " ", new_sentence)
        # new_sentence = re.sub("[^\w\s]", "", new_sentence)  # remove punc.

        if new_sentence != "":
            clean_sentence.append(new_sentence)
    return clean_sentence


def clean_tokenize(strings_list, combine):
    tokenized = []
    cleaned_string = preprocess_translate(strings_list)
    for s in cleaned_string:
        tokenized.append(nltk.word_tokenize(s))
    if combine:
        tokenized = np.concatenate(tokenized)

    # print(tokenized, tokenized.__len__())
    return tokenized


def frequency_distribution(tokenized_arr):
    if isinstance(tokenized_arr, list) and isinstance(tokenized_arr[0], list):
        freq = [dict(nltk.FreqDist(t)) for t in tokenized_arr]
        for e in freq:
            if "-" in e:
                del e["-"]
        return freq

    freq = dict(nltk.FreqDist(tokenized_arr))
    del freq["-"]
    return freq


# TEST CODE
if __name__ == "__main__":
    import pandas as pd

    training_data = pd.read_csv("../data/hns_2018_2019.csv", encoding="utf-8")
    stop_words = pd.read_csv("../data/stopwords.txt", encoding="utf-8")
    # print(data_set)
    # print(stop_words)
    # data_set.info()

    td_bckup = training_data

    training_data["Title"] = training_data["Title"].str.lower()
    training_data["Post Type"] = training_data["Post Type"].str.lower()

    # [print(i) for i in data_set["Title"]]
    # [print(i) for i in data_set["Post Type"]]
    # quit(-1)

    sentences_list = list(training_data["Title"])

    # timer_start = time.perf_counter()
    vocabulary = clean_tokenize(sentences_list, combine=True)
    vocabulary_freq = frequency_distribution(vocabulary)
    # time_taken = time.perf_counter() - timer_start
    # print("translating t1 : ", time_taken)

    # quit(-1)

    post_types_list = list(training_data["Post Type"])
    post_freq = nltk.FreqDist(post_types_list)

    # clean_tokenize_freq(sentences)
    #
    #
    # t1 = "terminal terminal terminal terminal terminal: ! #$%&\'()*+,-./:;<=>?@[\]^\_`{|}~  how the airport came to embody our national psychosis 213432324"
    # t2 = "not only is it possible to beat google, it could happen sooner"
    #
    # timer_start = time.perf_counter()
    # t1_a = preprocess_translate([t1])
    # print(t1_a)
    # time_taken = time.perf_counter() - timer_start
    # print("translating t1 : ", time_taken)
    #
    # timer_start = time.perf_counter()
    # t1_b = preprocess_regex([t1])
    # print(t1_b)
    # time_taken = time.perf_counter() - timer_start
    # print("regex t1 : ", time_taken)
    #
    # # import nltk
    #
    # words_tokenized = nltk.word_tokenize(t1_a[0])
    # print("tokenized", words_tokenized)
    # words_freq_dist = nltk.FreqDist(words_tokenized)
    # print("frequency distribution of words", words_freq_dist.items())

    # nltk.download('punkt')
    # nltk.download('wordnet')
    # print(nltk.word_tokenize(t1))

    pass
