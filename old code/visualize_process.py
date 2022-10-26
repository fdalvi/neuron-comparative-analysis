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
from sklearn import manifold, datasets

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

def compute(tag, layer, number):
    # X, y, mapping = utils.create_tensors(tokens, activations, 'NN',binarized_tag=tag)
    # X = X.reshape(-1,13,768)[:,layer,:]
    # label2idx, idx2label, src2idx, idx2src = mapping
    # # _, probeless_neurons,_ = probeless.get_neuron_ordering(X,y)
    # # probeless_neurons = probeless_neurons[:number]
    # probeless_neurons = np.loadtxt("neurons/probeless/" + tag + "/" + str(layer) + "_neurons.txt",dtype = int)
    # probeless_values = np.loadtxt("neurons/probeless/" + tag + "/" + str(layer) + "_values.txt")
    # weighted_neurons = np.loadtxt("neurons/weight/" + tag + "/" + str(layer) + "_neurons.txt",dtype=int)
    # weighted_values = np.loadtxt("neurons/weight/" + tag + "/" + str(layer) + "_values.txt")
    
    # rus = RandomUnderSampler(random_state=0)
    # X,y = rus.fit_resample(X, y)
    # # probeless_values = np.exp(probeless_values)
    # def train_dev_test_split(X,y,dev_ratio, test_ratio):
    #     index = np.arange(len(y))
    #     np.random.shuffle(index)
    #     y = y[index]
    #     X = X[index]
    #     ratio = 1 - dev_ratio - test_ratio
    #     train_len = int(ratio * len(y))
    #     test_len = int(test_ratio * len(y))
    #     dev_len = int(dev_ratio * len(y))
    #     train_len = len(y) - test_len - dev_len
    #     return X[:train_len], y[:train_len], X[train_len:train_len + dev_len], y[train_len:train_len + dev_len], X[train_len + dev_len: ], y[train_len + dev_len: ]
    # # train_test_split_ratio = 0.7
    # dev_ratio = 0.15
    # test_ratio = 0.15
    # X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(X,y, dev_ratio, test_ratio)
    X_train = np.load("data/train_data_"+ str(layer) + "_"+ tag + ".npy")
    X_dev = np.load("data/dev_data_"+ str(layer) + "_"+ tag + ".npy" )
    X_test = np.load("data/test_data_"+ str(layer) + "_"+ tag + ".npy" )
    y_train = np.load("data/train_label_" + tag + ".npy")
    y_test = np.load("data/test_label_" + tag+ ".npy")
    y_dev = np.load("data/dev_label_" + tag+ ".npy")
    probeless = np.loadtxt("neurons_splits/probeless/"+tag+"/"+str(layer)+"_neurons.txt",dtype = int)
    probeless_values = np.loadtxt("neurons_splits/probeless/"+tag+"/"+str(layer)+"_values.txt")
    selectivity = np.loadtxt("neurons_splits/sel/"+tag+"/"+str(layer)+"_neurons.txt",dtype = int)
    lca_elasticnet = np.loadtxt("neurons_splits/lca_elasticnet/"+tag+"/"+str(layer)+"_neurons.txt",dtype = int)
    lca_lasso = np.loadtxt("neurons_splits/lca_lasso/"+tag+"/"+str(layer)+"_neurons.txt",dtype = int)
    lca_ridge = np.loadtxt("neurons_splits/lca_ridge/"+tag+"/"+str(layer)+"_neurons.txt",dtype = int)
    n_components = 2
    lca = np.loadtxt("neurons_splits/lca/"+tag+"/"+str(layer)+"_neurons.txt",dtype = int)
    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X_train.T)  # 转换后的输出
    probeless_values = (probeless_values-np.min(probeless_values)) / (np.max(probeless_values) - np.min(probeless_values))
    return X_tsne, probeless# activations, num_layers = data_loader.load_activations('../activations_pos.json', 768)
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

tags = ["VBG","VBZ","NNPS","DT","TO","CD","JJ"]
l = [10,20,30,40,50,60,70,80,90,100]
lamda = [ 0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
os.makedirs("result_splits",exist_ok=True)
X,p = compute("NNPS",10,100)
import matplotlib.pyplot as plt

for i in range(200):
    c = np.zeros(768) + 1
    c[p[0:i]] = 0.2
    plt.scatter(X[:,0],X[:,1],20,c)
    plt.savefig("../figures/probeless_1_"+str(i))
    plt.cla()
# for t in tags:
#     result = []
#     for i in range(13):
#             # result.append(compute(t,i,[5,10,20,30,50,100]))
#         np.savetxt("result_splits/" + t + "_" + str(i) + "_splits" + ".txt", compute(t,i,l))
#     # result = np.array(result)
#     # np.savetxt("result/" + t +"_fuse_weight_probeless.txt", result)
