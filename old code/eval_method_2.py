import numpy as np
import numpy as np
import torch
import sys
sys.path.append("..")
import neurox.data.loader as data_loader
import neurox.interpretation.utils as utils
import neurox.interpretation.linear_probe as linear_probe
import neurox.interpretation.probeless as probeless
import neurox.data.extraction.transformers_extractor as transformers_extractor
from imblearn.under_sampling import RandomUnderSampler
import os
import neurox.interpretation.ablation as ablation
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def compute_best(settings, number, layer, tag):
    all_neurons = []
    for s in settings:
        neuron = np.loadtxt("neurons_splits/" + s + "/" + tag + "/" + str(layer) + "_neurons.txt",dtype = int)[:100]
        all_neurons.append(neuron)
    rank = combine_rankings(all_neurons)
    return rank

def eval_method(setting, number, layer, tag, neurons_best):
    neurons = np.loadtxt("neurons_splits/" + setting + "/" + tag + "/" + str(layer) + "_neurons.txt",dtype = int)[:100]
    ret1 = list(set(neurons[:number]).intersection(set(neurons_best[:number])))
    ret2 = list(set(neurons[:number]).union(set(neurons_best[:number])))
    s =  len(ret1) / len(ret2)
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
    ranks = np.zeros((768, ), dtype=int)
    for ranking_idx, ranking in enumerate(rankings):
        for idx, neuron in enumerate(ranking):
            rank = num_neurons - idx
            ranks[neuron] += rank

    # Sort neurons by their cumulative ranks and return in
    # descending order
    return np.argsort(ranks)[::-1].tolist()

tags = ["VBG","VBZ","NNPS","DT","TO","CD","JJ", "PRP","MD", "RB", "VBP", "VB", "NNS", "VBN", "POS", "IN", "NN", "CC", "NNP", "VBD"]
settings = ['gaussian_probe', 'lca_l2_001_l1_01_modif',  'lca_l1_01_modif',  'lca_l2_01_modif',  'probeless', 'sel', "iou_probe"]
axis = ['Gaussian',  'LCA',  "L1-.01", "L2-.01", "Probeless", "Sel", "IOU"]
print(settings)
l = [10,20,30,40,50,60,70,80,90,100]
layer = [0,1,2,3,4,5,6,7,8,9,10,11,12]
all_nums = []
score_all = []
for li in l:
    score = []
    
    for la in layer:
        score_layer = []
        for ta in tags:
            score_tag = []
            neurons_best = compute_best(settings, li, la, ta)
            for i in range(len(settings)):
                score_tag.append(eval_method(settings[i],li, la, ta, neurons_best))
            score_layer.append(score_tag)
        score.append(score_layer)
    score_all.append(score)
score_all=np.array(score_all)
all_res = np.mean(score_all, axis=2)
print(all_res.shape)

for i in range(all_res.shape[1]):
    np.savetxt("layer_" + str(i) + "score_2.txt", all_res[:,i,:])
# np.savetxt("score_2.npy", score_all)
