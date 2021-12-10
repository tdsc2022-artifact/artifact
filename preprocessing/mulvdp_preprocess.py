''' txt to pkl

transform txt-format codegadget to pkl-format for pytorch-lightning model
'''
from os import path
from utils.clean_gadget import clean_gadget
from sklearn.model_selection import train_test_split
import numpy
import os

from omegaconf import DictConfig
from models.mulvuldeepecker.buffered_path_context import BufferedPathContext
from utils.vectorize_gadget import GadgetVectorizer
import hashlib
from utils.unique import getMD5


def parse_file(cgd_txt: str):
    """
    Parses gadget file to find individual gadgets
    Yields each gadget as list of strings, where each element is code line
    Has to ignore first line of each gadget, which starts as integer+space
    At the end of each code gadget is binary value
        This indicates whether or not there is vulnerability in that gadget

    :param cgd_txt: code gadget in txt format
    :return:
    """
    with open(cgd_txt, "r", encoding="utf8") as file:
        gadget = []
        gadget_val = 0
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            if "-" * 33 in line and gadget:
                yield clean_gadget(gadget), gadget_val
                gadget = []
            elif stripped.split()[0].isdigit():
                if gadget:
                    # Code line could start with number (somehow)
                    if stripped.isdigit():
                        gadget_val = int(stripped)
                    else:
                        gadget.append(stripped)
            else:
                gadget.append(stripped)


def preprocess(config: DictConfig):
    '''
    key function

    '''
    holdout_data_path = path.join(
        config.data_folder,
        config.name,
        config.dataset.name,
        "all.txt",
    )
    if (not os.path.exists(holdout_data_path)):
        print(f"there is no file named: {holdout_data_path}")
        return
    output_train_path = path.join(config.data_folder, config.name,
                                  config.dataset.name, "train.pkl")
    if (os.path.exists(output_train_path)):
        return
    output_test_path = path.join(config.data_folder, config.name,
                                 config.dataset.name, "test.pkl")
    output_val_path = path.join(config.data_folder, config.name,
                                config.dataset.name, "val.pkl")
    vocab_path = path.join(config.data_folder, config.name,
                           config.dataset.name, "vocab.pkl")
    # if (os.path.exists(output_train_path)):
    #     print(f"{output_train_path} exists!")
    gadgets = dict()  # {md5:}
    count = 0
    dulCount = 0
    mulCount = 0
    vectorizer = GadgetVectorizer(config)
    for gadget, val in parse_file(holdout_data_path):
        count += 1
        print("Collecting gadgets...", count, end="\r")
        tokenized_gadget, backwards_slice = GadgetVectorizer.tokenize_gadget(
            gadget)
        tokenized_gadget_md5 = getMD5(str(tokenized_gadget))
        if (tokenized_gadget_md5 not in gadgets):
            row = {"gadget": gadget, "val": val, "count": 0}
            gadgets[tokenized_gadget_md5] = row
        else:
            if (gadgets[tokenized_gadget_md5]["val"] != -1):
                if (gadgets[tokenized_gadget_md5]["val"] != val):
                    dulCount += 1
                    gadgets[tokenized_gadget_md5]["val"] = -1
            mulCount += 1
        gadgets[tokenized_gadget_md5]["count"] += 1
    print("Find multiple...", mulCount)
    print("Find dulplicate...", dulCount)
    gadgets_unique = list()
    for gadget_md5 in gadgets:
        if (gadgets[gadget_md5]["val"] != -1):  # remove dulplicated
            vectorizer.add_gadget(gadgets[gadget_md5]["gadget"])
            gadgets_unique.append(gadgets[gadget_md5])
            # for i in range(gadgets[gadget_md5]["count"]):# do not remove mul
            #     vectorizer.add_gadget(gadgets[gadget_md5]["gadget"])
    print('Found {} forward slices and {} backward slices'.format(
        vectorizer.forward_slices, vectorizer.backward_slices))

    print("Training word2vec model...", end="\r")
    w2v_path = path.join(config.data_folder, config.name, config.dataset.name,
                         "w2v.model")
    vectorizer.train_model(w2v_path)
    vectorizer.build_vocab(vocab_path)
    global_vectors = []
    local_vectors = []
    labels = []
    count = 0
    for gadget in gadgets_unique:
        count += 1
        print("Processing gadgets...", count, end="\r")
        global_vector, backwards_slice = vectorizer.vectorize2(
            gadget["gadget"])  # [word len, embedding size]
        global_vectors.append(global_vector)
        local_vector = vectorizer.local_vectorize(gadget["gadget"],
                                                  config.sensi_api_path)
        local_vectors.append(local_vector)
        labels.append(gadget["val"])
    # numpy.random.seed(52)
    # numpy.random.shuffle(vectors)
    # numpy.random.seed(52)
    # numpy.random.shuffle(labels)
    # numpy.random.seed(52)
    global_vectors = numpy.array(global_vectors)
    local_vectors = numpy.array(local_vectors)
    labels = numpy.array(labels)
    positive_idxs = numpy.where(labels == 1)[0]
    negative_idxs = numpy.where(labels == 0)[0]
    # undersampled_negative_idxs = numpy.random.choice(negative_idxs,
    #                                                  len(positive_idxs),
    #                                                  replace=False)
    # resampled_idxs = numpy.concatenate(
    #     [positive_idxs, undersampled_negative_idxs])
    if len(positive_idxs) > len(positive_idxs):
        positive_idxs = numpy.random.choice(positive_idxs,
                                            len(negative_idxs),
                                            replace=False)
    elif len(positive_idxs) > len(positive_idxs):
        negative_idxs = numpy.random.choice(negative_idxs,
                                            len(positive_idxs),
                                            replace=False)
    else:
        pass
    resampled_idxs = numpy.concatenate([positive_idxs, negative_idxs])

    X_train, X_test, local_X_train, local_X_test, y_train, y_test = train_test_split(
        global_vectors[resampled_idxs],
        local_vectors[resampled_idxs],
        labels[resampled_idxs],
        test_size=0.2,
        stratify=labels[resampled_idxs])
    X_test, X_val, local_X_test, local_X_val, y_test, y_val = train_test_split(
        X_test, local_X_test, y_test, test_size=0.5, stratify=y_test)
    bpc = BufferedPathContext.create_from_lists(list(X_train),
                                                list(local_X_train),
                                                list(y_train))
    bpc.dump(output_train_path)
    bpc = BufferedPathContext.create_from_lists(list(X_test),
                                                list(local_X_test),
                                                list(y_test))
    bpc.dump(output_test_path)
    bpc = BufferedPathContext.create_from_lists(list(X_val), list(local_X_val),
                                                list(y_val))
    bpc.dump(output_val_path)
    return
