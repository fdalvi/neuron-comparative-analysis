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



def compute(set1, set2, number, tag, layer):
    # X, y, mapping = utils.create_tensors(tokens, activations, 'NN',binarized_tag=tag)
    # X = X.reshape(-1,13,768)[:,layer,:]
    # label2idx, idx2label, src2idx, idx2src = mapping
    # _, probeless_neurons,_ = probeless.get_neuron_ordering(X,y)
    # probeless_neurons = probeless_neurons[:number]
    sum_all = []
    neurons1 = neurons_dict[str(layer)+set1+tag][:number]
    neurons2 = neurons_dict[str(layer)+set2+tag][:number]
    ret1 = list(set(neurons1).intersection(set(neurons2)))
    ret2 = list(set(neurons1).union(set(neurons2)))
    s =  len(ret1) / len(ret2)

    return s

import os
tags = ["VBG","VBZ","NNPS","DT","TO","CD","JJ", "PRP","MD", "RB", "VBP", "VB", "NNS", "VBN", "POS", "IN", "NN", "CC", "NNP", "VBD"]
settings = ['gaussian_probe', 'lca_l2_001_l1_01_modif',  'lca_l1_01_modif',  'lca_l2_01_modif',  'probeless', 'sel', "iou_probe"]
axis = ['Gaussian',  'LCA',  "L1-.01", "L2-.01", "Probeless", "Sel", "IOU"]
neurons_dict = {}
for i in range(13):
    for j in range(len(settings)):
        for k in range(len(tags)):
            neurons_dict[str(i) + settings[j]+tags[k]] = np.loadtxt("neurons_splits/" + settings[j] + "/" + tags[k] + "/" + str(i) + "_neurons.txt",dtype = int)
l = [10,20,30,40,50,60,70,80,90,100]
os.makedirs("plot/",exist_ok=True)

layer = [0,1,2,3,4,5,6,7,8,9,10,11,12]
all_res = []
for li in l:
    all_nums = []
    for la in layer:
        all_nums_layer = [ ]
        for ta in tags:
            m = np.zeros((7,7))
            for i in range(7):
                for j in range(7):
                    m[i][j] = compute(settings[i],settings[j],li, ta, la)
                m[i][i] = 0.0
            compati = np.mean(m,axis=0)  
            all_nums_layer.append(compati)
        all_nums.append(all_nums_layer)
    #all_nums = np.array(all_nums)

    # all_nums = np.mean(all_nums, axis=1)
    all_res.append(all_nums)
all_res = np.array(all_res)
all_res = np.mean(all_res, axis=2)
print(all_res.shape)

for i in range(all_res.shape[1]):
    np.savetxt("layer_" + str(i) + "score_1.txt", all_res[:,i,:])
## [0.1074849  0.07918752 0.27588483 0.21961818 0.27467031 0.10792352
 ## 0.20842318 0.26455841 0.23707012 0.14389334]