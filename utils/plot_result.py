import numpy as np
import matplotlib.pyplot as plt
import json
from os import path
from utils.common import CWEID_ADOPT


def plot_approach_f1(name: str):
    '''
    :param name: approach name
    '''
    plt.rcParams['figure.figsize'] = (18, 12)
    plt.title(name)
    with open(path.join("res", f"{name}.json"), "r") as f:
        res = json.load(f)

        recall = list()
        precision = list()
        f1 = list()
        total_width, n = 0.6, 3
        width = total_width / n
        labels = list()
        for project in res:
            if not project in CWEID_ADOPT:
                continue
            labels.append(project)
            recall.append(res[project]["recall"])
            precision.append(res[project]["precision"])
            f1.append(res[project]["f1"])

        fig, ax = plt.subplots()
        # labels = np.array(labels)
        # recall = np.array(recall)
        # precision = np.array(precision)
        # f1 = np.array(f1)
        ax.set_title(name, fontsize=15)
        size = len(labels)
        x = np.arange(size)
        x = x - (total_width - width) / 2
        rects1 = ax.bar(x, precision, width=width, label='precision')
        rects2 = ax.bar(x + width,
                        recall,
                        width=width,
                        label='recal',
                        tick_label=labels)
        rects3 = ax.bar(x + 2 * width, f1, width=width, label='f1')
        plt.xticks(fontsize=15, rotation=20)

        ax.legend(loc='upper left', fontsize=12)
        autolabel(rects1, ax)
        autolabel(rects2, ax)
        autolabel(rects3, ax)

        out_dir = 'res/plot_png'
        plt.savefig(path.join(out_dir, name + ".png"))
        plt.close()
        f.close()


def plot_cwe_f1(cwe_id: str):
    plt.rcParams['figure.figsize'] = (18, 12)
    known_models = [
        "token", "code2vec", "code2seq", "vuldeepecker", "sysevr",
        "mulvuldeepecker", "vgdetector"
    ]
    recall = list()
    precision = list()
    f1 = list()
    labels = known_models
    size = len(labels)
    x = np.arange(size)
    total_width, n = 0.6, 3
    width = total_width / n
    x = x - (total_width - width) / 2
    for model in known_models:
        with open(path.join("res", f"{model}.json"), "r") as f:
            res = json.load(f)
            if cwe_id in list(res.keys()):
                recall.append(res[cwe_id]["recall"])
                precision.append(res[cwe_id]["precision"])
                f1.append(res[cwe_id]["f1"])
            else:
                recall.append(0)
                precision.append(0)
                f1.append(0)
            f.close()
    fig, ax = plt.subplots()
    ax.set_title(cwe_id, fontsize=15)
    rects1 = ax.bar(x, precision, width=width, label='precision')
    rects2 = ax.bar(x + width,
                    recall,
                    width=width,
                    label='recal',
                    tick_label=labels)
    rects3 = ax.bar(x + 2 * width, f1, width=width, label='f1')
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)
    plt.xticks(x + width, labels, rotation=20, fontsize=15)
    plt.legend(loc='upper left', fontsize=12)
    out_dir = 'res/plot_png'
    plt.savefig(path.join(out_dir, cwe_id + ".png"))
    plt.close()


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            '%.2f' % height,
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=12)
