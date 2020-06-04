import re
import time
import nltk
import string
import pandas as pd


def preprocess_regex(sentences):
    clean_sentence = []
    # for sentence in (sentences[:][0]):  # this is Df_pd for Df_np (text[:])
    for sentence in sentences:  # this is Df_pd for Df_np (text[:])
        new_sentence = re.sub("<.*?>", "", sentence)  # remove HTML tags
        new_sentence = re.sub("[^\w\s]", "", new_sentence)  # remove punc.
        new_sentence = re.sub("\d+", "", new_sentence)  # remove numbers
        # new_sentence = new_sentence.lower()  # lower case, .upper() for upper
        if new_sentence != "":
            clean_sentence.append(new_sentence)
    return clean_sentence


def preprocess_translate(sentences):
    clean_sentence = []
    excluded_punc = string.punctuation
    # for sentence in (sentences[:][0]):  # this is Df_pd for Df_np (text[:])
    for sentence in sentences:  # this is Df_pd for Df_np (text[:])
        new_sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        new_sentence = new_sentence.translate(str.maketrans('', '', string.digits))
        # new_sentence = re.sub(" +", " ", new_sentence)
        new_sentence = re.sub("[^\w\s]", "", new_sentence)  # remove punc.
        # new_sentence = re.sub("[^\s]", "", new_sentence)  # remove punc.

        if new_sentence != "":
            clean_sentence.append(new_sentence)
    return clean_sentence


# TEST CODE
if __name__ == "__main__":
    import pandas as pd

    data_set = pd.read_csv("../data/hns_2018_2019.csv", encoding="utf-8")
    stop_words = pd.read_csv("../data/stopwords.txt")
    # print(data_set)
    # print(stop_words)
    # data_set.info()

    data_set["Title"] = list(data_set["Title"].str.lower())

    [print(i) for i in data_set["Title"]]

    sentences = list(data_set["Title"])

    clean_sentences = preprocess_regex(sentences)

    t1 = "terminal terminal terminal terminal terminal: ! #$%&\'()*+,-./:;<=>?@[\]^\_`{|}~  how the airport came to embody our national psychosis 213432324"
    t2 = "not only is it possible to beat google, it could happen sooner"

    timer_start = time.perf_counter()
    t1_a = preprocess_translate([t1])
    print(t1_a)
    time_taken = time.perf_counter() - timer_start
    print("translating t1 : ", time_taken)

    timer_start = time.perf_counter()
    t1_b = preprocess_regex([t1])
    print(t1_b)
    time_taken = time.perf_counter() - timer_start
    print("regex t1 : ", time_taken)

    import nltk

    tokenized = nltk.word_tokenize(t1_a[0])
    print("tokenized", tokenized)
    freq_dist = nltk.FreqDist(tokenized)
    print("frequency distribution of words", freq_dist.items())

    # nltk.download('punkt')
    # nltk.download('wordnet')
    # print(nltk.word_tokenize(t1))

    pass
