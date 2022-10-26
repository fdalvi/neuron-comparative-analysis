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
import argparse
import neurox.interpretation.ablation as ablation
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def compute_best(args,settings, number, layer, tag):
    all_neurons = []
    for s in settings:
        neuron = np.loadtxt(args.input_folder + "/" + s + "/" + tag + "/" + str(layer) + "_neurons.txt",dtype = int)[:100]
        all_neurons.append(neuron)
    rank = combine_rankings(all_neurons)
    return rank

def eval_method(args, setting, number, layer, tag, neurons_best):
    neurons = np.loadtxt(args.input_folder + "/" + setting + "/" + tag + "/" + str(layer) + "_neurons.txt",dtype = int)[:100]
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




# np.savetxt("score_2.npy", score_all)
def main():
    
    parser = argparse.ArgumentParser(
            description="Extract Neurons")
    parser.add_argument('--input_folder', type=str,default='neurons',
                       help='folder contains raw data')
    parser.add_argument('--out_path', type=str, default='score',
                       help='Output path. Default to ./output/')
    parser.add_argument('--setting', type=str, default="LCA",
                       help='settings for extracting neurons', choices=["random", 'Noreg', 'Gaussian',  'LCA',  'Lasso-01',  'Ridge-01',  'Probeless', 'Selectivity', "IoU"])
    parser.add_argument('--tag', type=str, default="NN",
                       help='choice for tags', choices=["VBG","VBZ","NNPS","DT","TO","CD","JJ", "PRP","MD", "RB", "VBP", "VB", "NNS", "VBN", "POS", "IN", "NN", "CC", "NNP", "VBD"])
    parser.add_argument('--layer', type=int, default=0,
                       help='Choice of layers')
    args = parser.parse_args()   

    tags = ["VBG","VBZ","NNPS","DT","TO","CD","JJ", "PRP","MD", "RB", "VBP", "VB", "NNS", "VBN", "POS", "IN", "NN", "CC", "NNP", "VBD"]
    settings_source = ['Gaussian',  'LCA',  'Lasso-01',  'Ridge-01',  'Probeless', 'Selectivity', "IoU"]
    settings_target = ["random", 'Noreg', 'Gaussian',  'LCA',  'Lasso-01',  'Ridge-01',  'Probeless', 'Selectivity', "IoU"]
    neurons_dict = {}
    for i in range(13):
        for j in range(len(settings_source)):
            for k in range(len(tags)):
                neurons_dict[str(i) + settings_source[j]+tags[k]] = np.loadtxt(args.input_folder + "/" + settings_source[j] + "/" + tags[k] + "/" + str(i) + "_neurons.txt",dtype = int)
    for i in range(13):
        for j in range(len(settings_target)):
            for k in range(len(tags)):
                neurons_dict[str(i) + settings_target[j]+tags[k]] = np.loadtxt(args.input_folder + "/" + settings_target[j] + "/" + tags[k] + "/" + str(i) + "_neurons.txt",dtype = int)
    l = [10,20,30,40,50,60,70,80,90,100]
    os.makedirs(args.out_path, exist_ok=True)

    layer = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    all_nums = []
    score_all = []
    for li in l:
        score = []
        for la in layer:
            score_layer = []
            for ta in tags:
                score_tag = []
                neurons_best = compute_best(args,settings_source, li, la, ta)
                for i in range(len(settings_target)):
                    score_tag.append(eval_method(args,settings_target[i],li, la, ta, neurons_best))
                score_layer.append(score_tag)
            score.append(score_layer)
        score_all.append(score)
    score_all=np.array(score_all)
    all_res = np.mean(score_all, axis=2)
    print(all_res.shape)

    for i in range(all_res.shape[1]):
        np.savetxt(args.out_path + "/layer_" + str(i) + "score_2.txt", all_res[:,i,:])
if __name__ == "__main__":
    main()