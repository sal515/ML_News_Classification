# import re
# import time
import nltk
import string
import numpy as np
import pandas as pd
from datetime import datetime

excluded_list = """!"#$%&\()*+,:;<=>?@[\\]^`{|}~–—‐‑"""
included_list = """'"-_’/."""


def tokenize(untokenized_string):
    tweet_tokenizer = nltk.TweetTokenizer()
    # FIXME: Remove
    # return nltk.word_tokenize(s)
    return tweet_tokenizer.tokenize(untokenized_string)


def preprocess_translate(sentences):
    cleaned_sentence = []
    excluded_symbols = [c for c in excluded_list]
    for symbol in sentences:  # this is Df_pd for Df_np (text[:])
        symbols = symbol.translate(str.maketrans('', '', string.digits))
        symbols = symbols.translate(str.maketrans('', '', string.ascii_letters))
        symbols = symbols.replace(" ", "")

        for s in symbols:
            if s not in excluded_symbols and s not in included_list:
                excluded_symbols.append(s)

    exclude = "".join(str(e) for e in excluded_symbols)

    # FIXME: combined replace
    for sentence in sentences:  # this is Df_pd for Df_np (text[:])
        sentence = sentence.replace("’", "'")
        sentence = sentence.replace("–", "-")
        sentence = sentence.replace("—", "-")
        sentence = sentence.replace("‐", "-")
        new_sentence = sentence.translate(str.maketrans('', '', exclude))

        # FIXME: DELETE
        #     # new_sentence = new_sentence.translate(str.maketrans('', '', '0123456789'))
        #     # new_sentence = symbols.translate(str.maketrans('', '', string.punctuation))
        #     # new_sentence = new_sentence.translate(str.maketrans('', '', string.digits))
        #     # new_sentence = re.sub(" +", " ", new_sentence)
        #     # new_sentence = re.sub("[^\w\s]", "", new_sentence)  # remove punc.
        if new_sentence != "":
            cleaned_sentence.append(new_sentence)
        # cleaned_sentence.append(new_sentence)
    return cleaned_sentence, excluded_symbols


def clean_tokenize(strings_list, combine):
    tokenized = []
    cleaned_string_list, all_symbols_list = preprocess_translate(strings_list)

    for s in cleaned_string_list:
        tokenized.append(tokenize(s))

    # FIXME: DELETE
    # counter = 0
    # for s in cleaned_string_list:
    #     tokens = tokenize(s)
    #     tt = []
    #     for t in tokens:
    #         tt.append(t + " --> " + str(counter))
    #     tokenized.append(tt)
    #     counter += 1
    #     # tokenized.append(tokenize(s))

    if combine:
        tokenized = np.concatenate(tokenized)
    return tokenized, all_symbols_list


def frequency_distribution(tokenized_arr):
    if isinstance(tokenized_arr, list) and isinstance(tokenized_arr[0], list):
        freq = [dict(nltk.FreqDist(t)) for t in tokenized_arr]
        for e in freq:
            if "-" in e:
                # FIXME: "Add ' "
                # FIXME: "Add _ "
                # del e["-"]
                pass
        return freq

    freq = dict(nltk.FreqDist(tokenized_arr))
    # Fixme Re delete
    # del freq["-"]
    return freq


def extract_dataset(dataset_path, stopwords_path, filterBy="Created At", trainingKey="2018", testingKey="2019"):
    data = pd.read_csv(dataset_path, encoding="utf-8")

    for col_name in list(data.columns):
        if data[col_name].dtype not in ['int64', 'float64']:
            try:
                data[col_name] = data[col_name].str.lower()
            except:
                print(col_name, " was not of type string")

    data_categ = "data_cat"
    category = list(data[filterBy])
    extracted_category = [str(datetime.strptime(str(d), '%Y-%m-%d %H:%M:%S').year) for d in category]
    data[data_categ] = extracted_category

    training_set = data[data[data_categ].isin([trainingKey.lower()])]
    testing_set = data[data[data_categ].isin([testingKey.lower()])]

    stop_words = pd.read_csv(stopwords_path, encoding="utf-8")

    return training_set, testing_set, stop_words


def test_tokenizing():
    global data
    cleaned_sentences, symbols = preprocess_translate(sentences_list)
    vocabulary_sentences_test, all_excluded_vocabulary_sentences_test = clean_tokenize(sentences_list, combine=False)
    data = {"Original": sentences_list, "Cleaned": cleaned_sentences, "Tokenized": vocabulary_sentences_test}
    pd.DataFrame(data).to_csv("../test_org_clean.txt", "\t")


# TEST CODE
if __name__ == "__main__":
    debug = 0

    dataset_path = "../data/hns_2018_2019.csv"
    stopword_path = "../data/stopwords.txt"
    training_test_filter = "Created At"
    trainingKey = "2018"
    testingKey = "2019"
    vocabulary_col_name = "Title"

    training_set, testing_set, stopwords = extract_dataset(dataset_path, stopword_path, training_test_filter, trainingKey, testingKey)

    # timer_start = time.perf_counter()
    sentences_list = list(training_set[vocabulary_col_name])

    # FIXME: DELETE Test func
    # test_tokenizing()

    vocabulary, all_excluded_vocabulary = clean_tokenize(sentences_list, combine=True)

    vocabulary_freq = frequency_distribution(vocabulary)
    print("freq: ", vocabulary_freq)

    # FIXME
    data = {"Cleaned": vocabulary}
    pd.DataFrame(data).sort_values(by="Cleaned", ascending=True).to_csv("../test_cleaned.txt")

    # pd.DataFrame(data, columns=["Original", "New"]).to_csv("../test.csv")

    # post_types_list = list(training_set["Post Type"])
    # # post_freq = dict(nltk.FreqDist(post_types_list))
    # post_freq = nltk.FreqDist(post_types_list)

    # FIXME
    # Filter by "Post Type"
    # story_titles = training_data[training_data["Post Type"].isin(["story"])]["Title"]

    # story_titles_list = list(story_titles)
    # story_titles_tokenized = clean_tokenize(story_titles_list)

    # ====

    # show_hn_titles = training_data[training_data["Post Type"].isin(["show_hn"])]["Title"]
    # ask_hn_titles = training_data[training_data["Post Type"].isin(["ask_hn"])]["Title"]

    # ====

    if debug:
        # clean_tokenize_freq(sentences)
        #
        t1 = [
            "terminal terminal-terminal terminal terminal: ! #$%&\'()*+,-./:;<=>?@[\]^\_`{|}~  how the airport came to embody our national psychosis 213432324",
            "yoo"]
        # t2 = "not only is it possible to beat google, it could happen sooner"
        #
        # timer_start = time.perf_counter()
        # t1_a = preprocess_translate([t1])
        # print(t1_a)
        # t1_aa = clean_tokenize(t1_a, True)
        t1_aa = clean_tokenize(t1, True)
        print(t1_aa)
        t1_aaa = frequency_distribution(t1_aa)
        print(t1_aaa)
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
