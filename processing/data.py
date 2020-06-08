# import re
# import time
import string
import nltk
import numpy as np
import pandas as pd

excluded = """!"#$%&\()*+,:;<=>?@[\\]^`{|}~–—‐‑"""
included = """'"-_’/."""


def tokenize(untokenized_string):
    tweet_tokenizer = nltk.TweetTokenizer()
    # return nltk.word_tokenize(s)
    return tweet_tokenizer.tokenize(untokenized_string)


def preprocess_translate(sentences):
    cleaned_sentence = []
    all_symbols = [c for c in excluded]
    # excluded_punc = string.punctuation
    # for symbols in (sentences[:][0]):  # this is Df_pd for Df_np (text[:])
    for symbol in sentences:  # this is Df_pd for Df_np (text[:])
        # symbols = symbol.replace("’", "'")
        # symbols = symbols.replace("–", "-")
        # symbols = symbols.replace("—", "-")
        # symbols = symbols.replace("‐", "-")
        symbols = symbol.translate(str.maketrans('', '', string.digits))
        symbols = symbols.translate(str.maketrans('', '', string.ascii_letters))
        # print("symbols")
        symbols = symbols.replace(" ", "")
        # print(symbols)

        for s in symbols:
            if s not in all_symbols and s not in included:
                all_symbols.append(s)
                # print(s)

    exclude = "".join(str(e) for e in all_symbols)
    # print(exclude)

    for sentence in sentences:  # this is Df_pd for Df_np (text[:])
        sentence = sentence.replace("’", "'")
        sentence = sentence.replace("–", "-")
        sentence = sentence.replace("—", "-")
        sentence = sentence.replace("‐", "-")
        new_sentence = sentence.translate(str.maketrans('', '', exclude))
        #     # print("sentence")
        #
        #     # new_sentence = new_sentence.translate(str.maketrans('', '', '0123456789'))
        #     # new_sentence = symbols.translate(str.maketrans('', '', string.punctuation))
        #     # new_sentence = new_sentence.translate(str.maketrans('', '', string.digits))
        #     # new_sentence = re.sub(" +", " ", new_sentence)
        #     # new_sentence = re.sub("[^\w\s]", "", new_sentence)  # remove punc.
        if new_sentence != "":
            cleaned_sentence.append(new_sentence)
        # cleaned_sentence.append(new_sentence)
    return cleaned_sentence, all_symbols


def clean_tokenize(strings_list, combine):
    tokenized = []
    cleaned_string_list, all_symbols_list = preprocess_translate(strings_list)

    for s in cleaned_string_list:
        # tokenized.append(nltk.word_tokenize(s))
        tokenized.append(tokenize(s))

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
    del freq["-"]
    return freq


# TEST CODE
if __name__ == "__main__":
    debug = 0

    training_data = pd.read_csv("../data/hns_2018_2019.csv", encoding="utf-8")
    stop_words = pd.read_csv("../data/stopwords.txt", encoding="utf-8")

    td_bckup = training_data

    training_data["Title"] = training_data["Title"].str.lower()
    training_data["Post Type"] = training_data["Post Type"].str.lower()

    # [print(i) for i in data_set["Title"]]
    # [print(i) for i in data_set["Post Type"]]

    # timer_start = time.perf_counter()
    sentences_list = list(training_data["Title"])
    cleaned_sentences, symbols = preprocess_translate(sentences_list)
    # vocabulary = clean_tokenize(sentences_list, combine=True)

    vocabulary_sentences_test, all_excluded_vocabulary_sentences_test = clean_tokenize(sentences_list, combine=False)

    vocabulary, all_excluded_vocabulary = clean_tokenize(sentences_list, combine=True)

    # vocabulary_freq = frequency_distribution(vocabulary)
    # print("freq: ", vocabulary_freq)

    data = {"Cleaned": vocabulary}
    pd.DataFrame(data).sort_values(by="Cleaned", ascending=True).to_csv("../test_cleaned.txt")

    # print(
    #     f"sentence size: {sentences_list.__len__()}, cleaned: {cleaned_sentences.__len__()} tokenized: {vocabulary_sentences_test.__len__()}")

    data = {"Original": sentences_list, "Cleaned": cleaned_sentences, "Tokenized": vocabulary_sentences_test}
    pd.DataFrame(data).to_csv("../test_org_clean.txt", "\t")

    # pd.DataFrame(data, columns=["Original", "New"]).to_csv("../test.csv")

    post_types_list = list(training_data["Post Type"])
    # post_freq = dict(nltk.FreqDist(post_types_list))
    post_freq = nltk.FreqDist(post_types_list)

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
