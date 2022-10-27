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


def compute_best(args, settings, number, layer, tag):
    all_neurons = []
    for s in settings:
        neuron = np.loadtxt(
            os.path.join(args.input_folder, s, tag, str(layer) + "_neurons.txt"),
            dtype=int,
        )[:100]
        all_neurons.append(neuron)
    rank = combine_rankings(all_neurons)
    return rank


def eval_method(args, setting, number, layer, tag, neurons_best):
    neurons = np.loadtxt(
        os.path.join(args.input_folder, setting, tag, str(layer) + "_neurons.txt"),
        dtype=int,
    )[:100]
    ret1 = list(set(neurons[:number]).intersection(set(neurons_best[:number])))
    ret2 = list(set(neurons[:number]).union(set(neurons_best[:number])))
    s = len(ret1) / len(ret2)
    return s


def combine_rankings(rankings):
    if len(rankings) == 0:
        return rankings
    # Convert everything to Python lists
    _r = []
    for ranking in rankings:
        if isinstance(ranking, list):
            ranking = np.array(ranking)
        assert isinstance(ranking, np.ndarray)
        _r.append(ranking)
    rankings = _r

    # Make sure all rankings have same number of neurons
    num_neurons = rankings[0].shape[0]
    # for ranking_idx, ranking in enumerate(rankings):
    # assert num_neurons == ranking.shape[0], f"Ranking at index {ranking_idx} has differing number of neurons"
    # assert ranking.sum() == num_neurons * (num_neurons-1)/2

    # For every neuron, its "rank" with respect to a particular ranking
    # is defined by its position. The first neuron has a rank of `num_neurons`
    # and the last neuron has a rank of `1`. Add ranks of neurons across
    # all rankings
    ranks = np.zeros((768,), dtype=int)
    for ranking_idx, ranking in enumerate(rankings):
        for idx, neuron in enumerate(ranking):
            rank = num_neurons - idx
            ranks[neuron] += rank

    # Sort neurons by their cumulative ranks and return in
    # descending order
    return np.argsort(ranks)[::-1].tolist()


# np.savetxt("score_2.npy", score_all)
def main():

    parser = argparse.ArgumentParser(description="Neuron Vote Method")
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
        for j in range(len(baseline_methods)):
            for k in range(len(tags)):
                neurons_dict[str(i) + baseline_methods[j] + tags[k]] = np.loadtxt(
                    args.input_folder
                    + "/"
                    + baseline_methods[j]
                    + "/"
                    + tags[k]
                    + "/"
                    + str(i)
                    + "_neurons.txt",
                    dtype=int,
                )
    for i in range(13):
        for j in range(len(settings)):
            for k in range(len(tags)):
                neurons_dict[str(i) + settings[j] + tags[k]] = np.loadtxt(
                    args.input_folder
                    + "/"
                    + settings[j]
                    + "/"
                    + tags[k]
                    + "/"
                    + str(i)
                    + "_neurons.txt",
                    dtype=int,
                )
    os.makedirs(args.out_path, exist_ok=True)

    neuronvote_score = []
    for num in num_of_neurons:
        neuronvote_score_num = []
        for layer in layers:
            neuronvote_score_num_layer = []
            for tag in tags:
                neuronvote_score_num_layer_tag = []
                neurons_best = compute_best(args, baseline_methods, num, layer, tag)
                for i in range(len(settings)):
                    neuronvote_score_num_layer_tag.append(
                        eval_method(args, settings[i], num, layer, tag, neurons_best)
                    )
                neuronvote_score_num_layer.append(neuronvote_score_num_layer_tag)
            neuronvote_score_num.append(neuronvote_score_num_layer)
        neuronvote_score.append(neuronvote_score_num)
    neuronvote_score = np.array(neuronvote_score)
    np.save(os.path.join(args.out_path, "neuronvote.npy"), neuronvote_score)


if __name__ == "__main__":
    main()
