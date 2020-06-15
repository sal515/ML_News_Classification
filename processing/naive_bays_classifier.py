import math
import processing.common as common


def calculate_scores(test_vocabulary_freq, train_cls_list, train_cls_prob, model_df, log_base):
    """Score calculations"""
    cls_scores = []
    for words_freqs in test_vocabulary_freq:
        temp = []
        for cls in train_cls_list:
            score = math.log(train_cls_prob[cls], log_base)
            for word_freq in words_freqs.items():
                if word_freq[0] in model_df:
                    score += word_freq[1] * math.log(model_df[word_freq[0]][cls], log_base)
                    # print(w_f[0], " -> ", model[w_f[0]][cls])
            temp.append(score)
            # print(w_f_d, " -> ", score)
        cls_scores.append(temp)
    return cls_scores


def classify(train_cls_list, cls_scores):
    """Classification by classifier : Determine class for each test sentence in the test set"""
    classification_out_list = list(map(lambda p_list: train_cls_list[p_list.index(max(p_list))], cls_scores))
    return classification_out_list


def classify_and_generate_result(test_set, test_vocabulary_freq, train_cls_list, cls_scores, debug):
    classification_out_list = classify(train_cls_list, cls_scores)

    """ Adding columns for dataframe of test ouptut"""
    if "Title" in test_set:
        classification_dt = {"Sentences": [str(i).encode('utf-8') for i in test_set["Title"]]}
    else:
        raise Exception("Title column not found for testing dataset")
    if debug: classification_dt.update({"Cleaned Sentences": test_vocabulary_freq})
    classification_dt.update({"Classification": classification_out_list})
    for i in range(0, train_cls_list.__len__()):
        classification_dt.update({str(train_cls_list[i]): [x[i] for x in cls_scores]})
    if "Post Type" in test_set:
        classification_dt.update({"True Classification": list(test_set["Post Type"])})
    else:
        raise Exception("Post Type column not found for testing dataset")
    right_wrong_list = [("right" if x == y else "wrong") for x, y in
                        zip(classification_dt["Classification"], classification_dt["True Classification"])]
    classification_dt.update({"right_wrong": right_wrong_list})

    return classification_dt


def test_clean_tokenize_wrapper(
        data_set,
        vocabulary_col,
        excluded_list,
        included_list):
    """This function returns the frequency of words in vocabulary"""

    """List of titles/sentences in the dataset"""
    sentences = list(data_set[vocabulary_col])

    """Create list of vocabulary and excluded symbols"""
    vocabulary, excluded_symbols = common.clean_tokenize(
        sentences,
        excluded_list,
        included_list,
        combine=False)

    """Create list frequency of the words in the vocabulary """
    return common.frequency_distribution(vocabulary)
