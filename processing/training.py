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

    data_dict = {"word": list(unique_vocabulary)}
    class_frequencies = []
    class_probabilities = []
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

        class_frequencies.append(common.frequency_distribution(vocab))
        if excluded_vocab.__len__() == 0:
            excluded_vocab.append(temp_excluded_vocab)
            continue
        np.concatenate(excluded_vocab)
        excluded_vocab.append([i for i in temp_excluded_vocab if i not in excluded_vocab[0]])

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

    """Calculating class probabilities"""
    # FIXME something wrong with class freq - I think it is fixed <<<
    total_classes = common.calc_total_cls_entries(classes_freq)
    classes_prob = classes_freq.copy()
    for (k, v) in classes_freq.items():
        classes_prob[k] = v / total_classes

    return classes_prob, data_dict, np.concatenate(excluded_vocab), classes
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
    pass
