import sys
import numpy as np
import pandas as pd
import processing.common as common

# pd.set_option('mode.sim_interactive', True)
# pd.set_option('expand_frame_repr', True)
# pd.set_option('display.column_space', 2)
# pd.set_option('display.max_colwidth', sys.maxsize)
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

    class_frequencies = []
    class_probabilities = []
    cls_keys = []
    data_dict = {"word": list(unique_vocabulary)}

    """List of all classes"""
    classes = np.sort(np.array(list(classes_freq.keys())))

    """For every class, get the vocabulary and frequency of all all the words"""
    excluded_vocab = None
    for cls in classes:
        temp_sentences = list(data_set[data_set[classes_col].isin([cls])][vocabulary_col])
        vocab, excluded_vocab = common.clean_tokenize(
            temp_sentences,
            excluded_list,
            included_list,
            combine=True)
        class_frequencies.append(common.frequency_distribution(vocab))

    """Calculating probabilities with smoothing"""
    for i in range(0, classes.__len__()):
        tem_prob = dict()
        cls_freq = class_frequencies[i]
        for w in unique_vocabulary:
            if w not in cls_freq:
                tem_prob[w] = smoothing / (classes_freq[classes[i]] + (smoothing * unique_vocabulary.__len__()))
                continue
            tem_prob[w] = (cls_freq[w] + smoothing) / (
                    classes_freq[classes[i]] + (smoothing * unique_vocabulary.__len__()))
        class_probabilities.append(tem_prob)

    """"Creating Dataframe to save to txt file"""
    # FIXME: remove counter not needed?
    # data_dict = {"counter": list(range(0, unique_vocabulary.__len__())), "all_voc": list(unique_vocabulary)}
    for i in range(0, classes.__len__()):
        cls_keys.append(classes[i] + " freq")
        cls_keys.append(classes[i])
        data_dict[classes[i] + " freq"] = []
        data_dict[classes[i]] = []
        for w in unique_vocabulary:
            if w not in class_probabilities[i]:
                data_dict[classes[i]].append("-")
                data_dict[classes[i] + " freq"].append("-")
            else:
                data_dict[classes[i]].append(class_probabilities[i][w])

            if w not in class_frequencies[i]:
                data_dict[classes[i] + " freq"].append(smoothing)
                continue
            data_dict[classes[i] + " freq"].append(class_frequencies[i][w])

    """Calculating class probabilites"""
    # FIXME something wrong with class freq - I think it is fixed <<<
    total_classes = common.calc_total_cls_entries(classes_freq)
    classes_prob = classes_freq.copy()
    for (k, v) in classes_freq.items():
        classes_prob[k] = v / total_classes

    return  classes_prob, data_dict, excluded_vocab, classes
    # FIXME : Check if i need to return all these
    # return class_frequencies, class_probabilities, classes_prob, data_dict, excluded_vocab, classes, cls_keys


def generate_model_df(training_data):
    """Conditional probability table as dictionary to easily access the probabilities"""
    if "word" in training_data:
        model_df = pd.DataFrame(training_data, index=training_data["word"]).to_dict(orient="index")
    else:
        raise Exception("Error: word column is not found in training data")
    return model_df


# TEST CODE
if __name__ == "__main__":
    # debug = 0
    #
    # """Constant variables"""
    # dataset_path = "../data/hns_2018_2019.csv"
    # stop_words_path = "../data/stopwords.txt"
    # data_filter = "Created At"
    # train_key = "2018"
    # test_Key = "2019"
    # vocabulary_col = "Title"
    # classes_col = "Post Type"
    # smoothing = 0.5
    # csv_path = "../output/task1.csv"
    # text_path = "../output/model-2018.txt"
    # vocabulary_path = "../output/vocabulary.txt"
    # removed_word_path = "../output/removed_word.txt"
    #
    # """Extracting data from the dataset files"""
    # train_set, test_set, train_cls_freq, test_cls_freq, stopwords = common.extract_dataset(dataset_path,
    #                                                                                 stop_words_path,
    #                                                                                 data_filter,
    #                                                                                 train_key, test_Key,
    #                                                                                 classes_col)
    #
    # """Get all vocabulary and frequency of all the words in TRAIN dataset"""
    # train_unique_vocabulary, train_vocabulary_freq = clean_tokenize_freq_dist(train_set, vocabulary_col, True)
    #
    # # FIXME testing data extraction not working
    # """Get all vocabulary and frequency of all the words in TEST dataset"""
    # # test_unique_vocabulary, test_vocabulary_freq = clean_tokenize_freq_dist(train_set, vocabulary_col, False)
    #
    # """Calculate conditional probabilities for each word in every class"""
    # train_cls_word_freq, train_cls_word_prob, train_cls_prob, training_data, excluded_vocab = generate_model(
    #     train_unique_vocabulary,
    #     train_set,
    #     classes_col,
    #     vocabulary_col,
    #     train_cls_freq,
    #     smoothing)
    #
    # """Store probabilities data frame to file"""
    # common.store_dataframe_to_file(training_data, csv_path=csv_path, text_path=text_path)
    # """Store vocabulary data frame to file"""
    # common.store_dataframe_to_file({"vocabulary": list(train_unique_vocabulary)}, csv_path=None, text_path=vocabulary_path)
    # """Store excluded data frame to file"""
    # common.store_dataframe_to_file({"removed": [str(i).encode('utf-8') for i in excluded_vocab]}, csv_path=None,
    #                         text_path=removed_word_path)
    #
    # # TODO: Text files: vocabulary and removed words and return carriage
    #
    # # FIXME: DELETE Test func
    # # test_tokenizing()
    # # print("freq: ", vocabulary_freq)
    # # data = {"Cleaned": vocabulary}
    # # pd.DataFrame(data).sort_values(by="Cleaned", ascending=True).to_csv("../test_cleaned.txt")
    # # pd.DataFrame(data, columns=["Original", "New"]).to_csv("../test.csv")
    # # timer_start = time.perf_counter()
    pass
