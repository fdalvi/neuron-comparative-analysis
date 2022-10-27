import sys

import numpy as np
import torch

sys.path.append(".")
import argparse
import os

import matplotlib.pyplot as plt
import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as data_loader
import neurox.interpretation.ablation as ablation
import neurox.interpretation.linear_probe as linear_probe
import neurox.interpretation.probeless as probeless
import neurox.interpretation.utils as utils
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics.pairwise import cosine_similarity


def compute(set1, set2, number, tag, layer, neurons_dict):

    neurons1 = neurons_dict[str(layer) + set1 + tag][:number]
    neurons2 = neurons_dict[str(layer) + set2 + tag][:number]
    ret1 = list(set(neurons1).intersection(set(neurons2)))
    ret2 = list(set(neurons1).union(set(neurons2)))
    s = len(ret1) / len(ret2)

    return s


def main():

    parser = argparse.ArgumentParser(description="Extract Neurons")
    parser.add_argument(
        "--input_folder", type=str, default="neurons", help="folder contains raw data"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="score",
        help="Output path. Default to ./output/",
    )
    parser.add_argument(
        "--settings", type=str, default="LCA", help="settings for extracting neurons"
    )
    parser.add_argument(
        "--baseline_methods",
        type=str,
        default="LCA,Probeless",
        help="settings for extracting neurons",
    )
    parser.add_argument("--tags", type=str, default="NN", help="choice for tags")
    parser.add_argument("--layers", type=str, default="0", help="Choice of layers")
    parser.add_argument(
        "--num_of_neurons", type=str, default=None, help="Choice of num of neurons"
    )

    args = parser.parse_args()

    tags = args.tags.split(",")
    settings = args.settings.split(",")
    baseline_methods = args.baseline_methods.split(",")
    num_of_neurons = args.num_of_neurons.split(",")
    for i in range(len(num_of_neurons)):
        num_of_neurons[i] = int(num_of_neurons[i])
    layers = args.layers.split(",")
    for i in range(len(layers)):
        layers[i] = int(layers[i])

    neurons_dict = {}
    for i in range(13):
        for j in range(len(settings)):
            for k in range(len(tags)):
                neurons_dict[str(i) + settings[j] + tags[k]] = np.loadtxt(
                    os.path.join(
                        args.input_folder, settings[j], tags[k], str(i) + "_neurons.txt"
                    ),
                    dtype=int,
                )

    for i in range(13):
        for j in range(len(baseline_methods)):
            for k in range(len(tags)):
                neurons_dict[str(i) + settings[j] + tags[k]] = np.loadtxt(
                    os.path.join(
                        args.input_folder,
                        baseline_methods[j],
                        tags[k],
                        str(i) + "_neurons.txt",
                    ),
                    dtype=int,
                )
    os.makedirs(args.out_path, exist_ok=True)

    avgoverlap_score = []
    for num in num_of_neurons:
        avgoverlap_score_nums = []
        for layer in layers:
            avgoverlap_score_nums_layer = []
            for tag in tags:
                overlap = np.zeros((len(baseline_methods), len(settings)))
                for i in range(overlap.shape[0]):
                    for j in range(overlap.shape[1]):
                        overlap[i][j] = compute(
                            baseline_methods[i],
                            settings[j],
                            num,
                            tag,
                            layer,
                            neurons_dict,
                        )
                avgoverlap_score_nums_layer_tag = np.mean(overlap, axis=0)
                avgoverlap_score_nums_layer.append(avgoverlap_score_nums_layer_tag)
            avgoverlap_score_nums.append(avgoverlap_score_nums_layer)
        avgoverlap_score.append(avgoverlap_score_nums)
    avgoverlap_score = np.array(avgoverlap_score)
    np.save(os.path.join(args.out_path, "avgoverlap.npy"), avgoverlap_score)


if __name__ == "__main__":
    main()
