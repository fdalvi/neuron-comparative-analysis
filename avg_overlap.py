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

import argparse

def compute(set1, set2, number, tag, layer, neurons_dict):
    neurons1 = neurons_dict[str(layer)+set1+tag][:number]
    neurons2 = neurons_dict[str(layer)+set2+tag][:number]
    ret1 = list(set(neurons1).intersection(set(neurons2)))
    ret2 = list(set(neurons1).union(set(neurons2)))
    s =  len(ret1) / len(ret2)

    return s


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
    os.makedirs(args.out_path ,exist_ok=True)

    layer = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    all_res = []
    for li in l:
        all_nums = []
        for la in layer:
            all_nums_layer = [ ]
            for ta in tags:
                m = np.zeros((len(settings_source),len(settings_target)))
                for i in range(m.shape[0]):
                    for j in range(m.shape[1]):
                        m[i][j] = compute(settings_source[i],settings_target[j],li, ta, la, neurons_dict)
                compati = np.mean(m,axis=0)  
                all_nums_layer.append(compati)
            all_nums.append(all_nums_layer)
        
        all_res.append(all_nums)
    all_res = np.array(all_res)
    all_res = np.mean(all_res, axis=2)
    for i in range(all_res.shape[1]):
        np.savetxt(args.out_path + "/layer_" + str(i) + "score_1.txt", all_res[:,i,:])
if __name__ == "__main__":
    main()