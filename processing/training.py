import sys
import numpy as np
import pandas as pd
import processing.common as common

# pd.set_option('mode.sim_interactive', True)
# pd.set_option('expand_frame_repr', True)
# pd.set_option('display.column_space', 2)
# pd.set_option('display.max_colwidth', sys.maxsize)
# from processing.common import clean_tokenize, frequency_distribution

pd.set_option('display.max_columns', sys.maxsize)
pd.set_option('display.max_rows', sys.maxsize)
pd.set_option('display.width', sys.maxsize)


def generate_model(
        unique_vocabulary,
        data_set,
        classes_freq,
        classes_col,
        vocabulary_col,
        excluded_list,
        included_list,
        smoothing):
    data_dict = {"word": list(unique_vocabulary)}
    class_word_probabilities = []
    class_word_frequencies = []
    excluded_vocab = []
    cls_keys = []

    """List of all classes"""
    classes = np.sort(np.array(list(classes_freq.keys())))

    """For every class, get the vocabulary and frequency of all all the words"""
    for cls in classes:
        temp_sentences = list(data_set[data_set[classes_col].isin([cls])][vocabulary_col])
        vocab, temp_excluded_vocab = common.clean_tokenize(
            temp_sentences,
            excluded_list,
            included_list,
            combine=True)

        class_word_frequencies.append(common.frequency_distribution(vocab))

        if excluded_vocab.__len__() == 0:
            excluded_vocab.append(temp_excluded_vocab)
            continue
        np.concatenate(excluded_vocab)
        excluded_vocab.append([i for i in temp_excluded_vocab if i not in excluded_vocab[0]])

    """Calculating probabilities with smoothing"""
    for i in range(0, classes.__len__()):
        tem_prob = dict()
        cls_w_freq = class_word_frequencies[i]
        for w in unique_vocabulary:
            if w not in cls_w_freq:
                tem_prob[w] = smoothing / (classes_freq[classes[i]] + (smoothing * unique_vocabulary.__len__()))
                continue
            tem_prob[w] = (cls_w_freq[w] + smoothing) / (
                    classes_freq[classes[i]] + (smoothing * unique_vocabulary.__len__()))
        class_word_probabilities.append(tem_prob)

    """"Creating Dataframe to save to txt file"""
    for i in range(0, classes.__len__()):
        cls_keys.append(classes[i] + " freq")
        cls_keys.append(classes[i])
        data_dict[classes[i] + " freq"] = []
        data_dict[classes[i]] = []
        for w in unique_vocabulary:
            if w not in class_word_probabilities[i]:
                data_dict[classes[i]].append("-")
                data_dict[classes[i] + " freq"].append("-")
            else:
                data_dict[classes[i]].append(class_word_probabilities[i][w])

            if w not in class_word_frequencies[i]:
                data_dict[classes[i] + " freq"].append(smoothing)
                continue
            data_dict[classes[i] + " freq"].append(class_word_frequencies[i][w])

    """Calculating each class probabilities"""
    total_classes = common.calc_total_cls_entries(classes_freq)
    classes_prob = classes_freq.copy()
    for (k, v) in classes_freq.items():
        classes_prob[k] = v / total_classes

    return classes_prob, data_dict, np.concatenate(excluded_vocab), classes


def generate_model_df(training_data):
    """Conditional probability table as dictionary to easily access the probabilities"""
    if "word" in training_data:
        model_df = pd.DataFrame(training_data, index=training_data["word"]).to_dict(orient="index")
    else:
        raise Exception("Error: word column is not found in training data")
    return model_df


def train_clean_tokenize_wrapper(
        data_set,
        vocabulary_col,
        excluded_list,
        included_list,
        trainType,
        stopwords,
        minWords,
        maxWords):
    """This function returns the unique words in vocabulary"""

    """List of titles/sentences in the dataset"""
    sentences = list(data_set[vocabulary_col])

    """Create list of vocabulary and excluded symbols"""
    vocabulary, excluded_symbols = common.clean_tokenize(
        sentences,
        excluded_list,
        included_list,
        combine=True)

    # FIXME: Remember to handle Train call of this function
    # FIXME: Test uses unique_vocab. -> uses vocabulary_freq
    # FIXME: 1. Stop words filtering - generate unique_vocab. by filtering the stop words here

    """Create list frequency of the words in the vocabulary """
    vocabulary_freq = common.frequency_distribution(vocabulary)

    """Removing stop words from the vocabulary"""
    if trainType == "stopwords":
        for word in stopwords:
            if word in vocabulary_freq:
                del vocabulary_freq[word]

    # FIXME: 2. Word length filtering - generate unique_vocab. by filtering the words outside the len size
    """Removing words with out of range length from the vocabulary"""
    if trainType == "word_length":
        to_be_removed = list(filter(lambda x: len(x) <= minWords or len(x) >= maxWords, vocabulary_freq.keys()))

        for word in to_be_removed:
            if word in vocabulary_freq:
                del vocabulary_freq[word]

    # FIXME: 3. Infrequent word filtering - ??

    return np.sort(np.array(list(vocabulary_freq.keys())))

    # FIXME: 3. Infrequent word filtering - ??

    # FIXME: Remember to handle Test call of this function
    # FIXME: Test uses vocab_freq


# TEST CODE
if __name__ == "__main__":
    pass
