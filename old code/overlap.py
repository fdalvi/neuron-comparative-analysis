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
# transformers_extractor.extract_representations('bert-base-uncased',
#     '../Data_for_Yimin/Data_for_Yimin/POS/sample.word.txt',
#     'activations_sample_pos.json',
#     device="cpu",
#     aggregation="average" #last, first
# )
# activations, num_layers = data_loader.load_activations('activations_sample_pos.json', 768)
# tokens = data_loader.load_data('../Data_for_Yimin/Data_for_Yimin/POS/sample.word.txt',
#                                 '../Data_for_Yimin/Data_for_Yimin/POS/sample.label.txt',
#                                 activations,
#                                 512 # max_sent_l
#                                 )
def plot(m,li,la,ta):
    fig, ax = plt.subplots()
    # im = ax.imshow(m)
    ax.set_xticks(np.arange(len(axis)), labels=axis)
    ax.set_yticks(np.arange(len(axis)), labels=axis)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    for p in range(11):
        for q in range(11):
            text = ax.text(q, p, round(m[p][q],2),
                        ha="center", va="center", color="w",size=8)
    im = ax.imshow(m)

    ax.set_title("Heatmap Layer " + str(la) + "tag: " +  ta + " neurons: " + str(li))
    fig.tight_layout()
    fig.savefig("plot_map_6/heatmap_layer" + str(la) + "_" + ta + "_neuron_" + str(li) + ".png")
    plt.cla()
# def compute(set1, set2, number, tags, layer):
#     # X, y, mapping = utils.create_tensors(tokens, activations, 'NN',binarized_tag=tag)
#     # X = X.reshape(-1,13,768)[:,layer,:]
#     # label2idx, idx2label, src2idx, idx2src = mapping
#     # _, probeless_neurons,_ = probeless.get_neuron_ordering(X,y)
#     # probeless_neurons = probeless_neurons[:number]
#     sum_all = []
#     for i in tags:
#         neurons1 = np.loadtxt("neurons_splits/" + set1 + "/" + i + "/" + str(layer) + "_neurons.txt",dtype = int)[:number]
#         neurons2 = np.loadtxt("neurons_splits/" + set2 + "/" + i + "/" + str(layer) + "_neurons.txt",dtype = int)[:number] 
#         ret1 = list(set(neurons1).intersection(set(neurons2)))
#         ret2 = list(set(neurons1).union(set(neurons2)))
#         s =  len(ret1) / len(ret2)
#     sum_all.append(s)

#     return np.mean(sum_all)
def compute(set1, set2, number, tag, layer):
    # X, y, mapping = utils.create_tensors(tokens, activations, 'NN',binarized_tag=tag)
    # X = X.reshape(-1,13,768)[:,layer,:]
    # label2idx, idx2label, src2idx, idx2src = mapping
    # _, probeless_neurons,_ = probeless.get_neuron_ordering(X,y)
    # probeless_neurons = probeless_neurons[:number]
    sum_all = []
    neurons1 = np.loadtxt("neurons_splits/" + set1 + "/" + tag + "/" + str(layer) + "_neurons.txt",dtype = int)[:number]
    neurons2 = np.loadtxt("neurons_splits/" + set2 + "/" + tag + "/" + str(layer) + "_neurons.txt",dtype = int)[:number] 
    ret1 = list(set(neurons1).intersection(set(neurons2)))
    ret2 = list(set(neurons1).union(set(neurons2)))
    s =  len(ret1) / len(ret2)

    return s
# activations, num_layers = data_loader.load_activations('../activations_pos.json', 768)
# tokens = data_loader.load_data('../Data_for_Yimin/Data_for_Yimin/POS/wsj.20.test.conllx.word',
#                                 '../Data_for_Yimin/Data_for_Yimin/POS/wsj.20.test.conllx.label',
#                                 activations,
#                                 512 # max_sent_l
#                                 )
# activations, num_layers = data_loader.load_activations('activations_sample_pos.json', 768)
# tokens = data_loader.load_data('../Data_for_Yimin/Data_for_Yimin/POS/sample.word.txt',
#                                 '../Data_for_Yimin/Data_for_Yimin/POS/sample.label.txt',
#                                 activations,
#                                 512 # max_sent_l
#                                 )
import os
tags = ["VBG","VBZ","NNPS","DT","TO","CD","JJ", "PRP","MD", "RB", "VBP", "VB", "NNS", "VBN", "POS", "IN", "NN", "CC", "NNP", "VBD"]
settings = ['gaussian_probe', 'lca_l2_0_l1_0_modif', 'lca_l2_001_l1_01_modif', 'lca_l1_001_modif', 'lca_l1_01_modif',  'lca_l2_001_modif', 'lca_l2_01_modif',  'probeless', 'sel', "iou_probe", "random1", "random2", "random3"]
axis = ['Gaussian', "NoReg", 'LCA', "L1-.001", "L1-.01", "L2-.001", "L2-.01", "Probeless", "Sel", "IOU", "Random"]
print(settings)
l = [10]
os.makedirs("plot/",exist_ok=True)

layer = [0,1,6,12]
for li in l:
    for la in layer:
        for ta in tags:
            m = np.zeros((13,13))
            for i in range(13):
                for j in range(13):
                    m[i][j] = compute(settings[i],settings[j],li, ta, la)
            n = np.zeros((11,11))
            n[0:10,0:10] = m[0:10,0:10]
            n[10,0:10] = np.mean(m[10:13,0:10], axis=0)
            n[0:10,10] = np.mean(m[0:10, 10:13], axis=1)
            n[10,10] = 1.0 
            plot(n,li,la,ta)
# for li in l:
#     for la in layer:
#         m = np.zeros((13,13))
#         for i in range(13):
#             for j in range(13):
#                 m[i][j] = compute(settings[i],settings[j],li, tags, la)
#         n = np.zeros((11,11))
#         n[0:10,0:10] = m[0:10,0:10]
#         n[10,0:10] = np.mean(m[10:13,0:10], axis=0)
#         n[0:10,10] = np.mean(m[0:10, 10:13], axis=1)
#         n[10,10] = 1.0 
#         plot(n,li,tags, la)

# li = 40
# ta = "JJ"
# la = 11
# m = np.zeros((16,16))
# for i in range(16):
#     for j in range(16):
#         m[i][j] = compute(settings[i],settings[j],li, ta, la)
# n = np.zeros((14,14))
# n[0:13,0:13] = m[0:13,0:13]
# n[13,0:13] = np.mean(m[13:16,0:13], axis=0)
# n[0:13,13] = np.mean(m[0:13, 13:16], axis=1)
# n[13,13] = 1.0 
# plot(n,li,la,ta)


# for li in l:
#     print(li)
#     for i in range(len(settings)):
#         for j in range(i, len(settings)):
#             print(settings[i] + "  " + settings[j] + "  " + str(compute(settings[i],settings[j], li)))
# tags = ["VBG","VBZ","NNPS","DT","TO","CD","JJ"]
# settings = os.listdir("neurons_splits/gaussian_probe")

# files = os.listdir("neurons_splits/gaussian_probe")
# for f in files:
#     if ".txt" in f:
#         h = f.split("_")
#         os.rename("neurons_splits/gaussian_probe/" + f,"neurons_splits/gaussian_probe/" + h[0] + "/" + h[1] + "_" + h[2] )
# l = [10,20,30,40,50,60,70,80,90,100]
# os.makedirs("result",exist_ok=True)
# r = []
# for t in tags:
#     result = []
#     for i in range(13):
#         x = []
#         for ll in l:
#             x.append(compute(t,i,ll))
#         result.append(x)
#     r.append(result)
# r = np.array(r)
# print(r.shape)
# r = np.mean(r,axis=0)
# print(r.shape)
# for i in range(r.shape[0]):
#     print("Layer: " + str(i))
#     for j in range(r.shape[1]):
#         print(str(r[i][j][0]) + " " + str(r[i][j][1]))
        # np.savetxt("result/" + t + "_" + str(i) + "_10_fuse_weight_probeless.txt", compute(t,i,l))
    # result = np.array(result)
    # np.savetxt("result/" + t +"_fuse_weight_probeless.txt", result)
