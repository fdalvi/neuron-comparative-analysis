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
from scipy.spatial import ConvexHull
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

def compute(tag, layer, number, name):
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

    # probeless = np.loadtxt("neurons_splits/probeless/"+tag+"/"+str(layer)+"_neurons.txt",dtype = int)
    # selectivity = np.loadtxt("neurons_splits/sel/"+tag+"/"+str(layer)+"_neurons.txt",dtype = int)
    # lca_elasticnet = np.loadtxt("neurons_splits/lca_elasticnet/"+tag+"/"+str(layer)+"_neurons.txt",dtype = int)
    lca= np.loadtxt("neurons_splits/" + name + "/"+tag+"/"+str(layer)+"_neurons.txt",dtype = np.int32)
    # lca_ridge = np.loadtxt("neurons_splits/lca_ridge_001/"+tag+"/"+str(layer)+"_neurons.txt",dtype = np.int32)
    # lca = np.loadtxt("neurons_splits/lca/"+tag+"/"+str(layer)+"_neurons.txt",dtype = int)
    def train_dev_and_eval(X_train, y_train, X_dev, y_dev, X_test, y_test, selected):
        X_train_sel = ablation.filter_activations_keep_neurons(X_train, selected)
        X_dev_sel = ablation.filter_activations_keep_neurons(X_dev, selected)
        X_test_sel =  ablation.filter_activations_keep_neurons(X_test, selected)
        # X_train_sel = torch.tensor(X_train_sel)
        # y_train = torch.tensor(y_train, dtype = torch.long)
        probe = linear_probe.train_logistic_regression_probe(X_train_sel, y_train, lambda_l1=0.00, lambda_l2=0.00)
        scores_selected = linear_probe.evaluate_probe(probe, X_test_sel, y_test)["__OVERALL__"]
        scores_selected_dev = linear_probe.evaluate_probe(probe, X_dev_sel, y_dev)["__OVERALL__"]
        return scores_selected, scores_selected_dev
    result = []
 
    for n in number:
        # probeless_sel = probeless[:n]
        # probeless_score, probeless_score_dev = train_dev_and_eval(X_train, y_train,X_dev, y_dev, X_test, y_test, probeless_sel)
        # selectivity_sel =selectivity[:n]
        # selectivity_score, selectivity_score_dev = train_dev_and_eval(X_train, y_train,X_dev, y_dev, X_test, y_test, selectivity_sel)
            

        # lca_elasticnet_sel = lca_elasticnet[:n]
        # lca_elasticnet_score, lca_elasticnet_score_dev = train_dev_and_eval(X_train, y_train,X_dev, y_dev, X_test, y_test, lca_elasticnet_sel)
        lca_sel = lca[:n]
        lca_score, lca_score_dev = train_dev_and_eval(X_train, y_train,X_dev, y_dev, X_test, y_test, lca_sel)
            

        # lca_ridge_sel = lca_ridge[:n]
        # lca_ridge_score, lca_ridge_score_dev = train_dev_and_eval(X_train, y_train,X_dev, y_dev, X_test, y_test, lca_ridge_sel)
        # lca_sel = lca[:n]
        # lca_score, lca_score_dev = train_dev_and_eval(X_train, y_train,X_dev, y_dev, X_test, y_test, lca_sel)
        # result.append([probeless_score, probeless_score_dev])
        # result.append([selectivity_score, selectivity_score_dev])
        # result.append([lca_elasticnet_score, lca_elasticnet_score_dev])
        result.append([lca_score, lca_score_dev])
        # result.append([lca_ridge_score, lca_ridge_score_dev])
        # result.append([lca_score, lca_score_dev])
            
    return result

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
tags = ["VBG","VBZ","NNPS","DT","TO","CD","JJ"]
# tags = ["PRP","MD", "RB", "VBP", "VB", "NNS", "VBN", "POS", "IN", "NN", "CC", "NNP", "VBD"]
l = [10,20,30,40,50,60,70,80,90,100]
os.makedirs("result_splits",exist_ok=True)

for t in tags:
    result = []
    for i in range(13):
            # result.append(compute(t,i,[5,10,20,30,50,100]))
        np.savetxt("result_splits/" + t + "_" + str(i) + "_probeless" + ".txt", compute(t,i,l, "probeless"))
        # np.savetxt("result_splits/" + t + "_" + str(i) + "_iou" + ".txt", compute(t,i,l , "iou_probe"))
        # np.savetxt("result_splits/" + t + "_" + str(i) + "_lca_l2_001_l1_01_modif" + ".txt", compute(t,i,l , "lca_l2_001_l1_01_modif"))
        # np.savetxt("result_splits/" + t + "_" + str(i) + "_lca_l1_01_modif" + ".txt", compute(t,i,l , "lca_l1_01_modif"))
        # np.savetxt("result_splits/" + t + "_" + str(i) + "_lca_l2_01_modif" + ".txt", compute(t,i,l , "lca_l2_01_modif"))
        # np.savetxt("result_splits/" + t + "_" + str(i) + "_lca_l1_001_modif" + ".txt", compute(t,i,l , "lca_l1_001_modif"))
        # np.savetxt("result_splits/" + t + "_" + str(i) + "_lca_l2_001_modif" + ".txt", compute(t,i,l , "lca_l2_001_modif"))
        # np.savetxt("result_splits/" + t + "_" + str(i) + "_lca_l2_0_l1_0_modif" + ".txt", compute(t,i,l , "lca_l2_0_l1_0_modif"))

    # result = np.array(result)
    # np.savetxt("result/" + t +"_fuse_weight_probeless.txt", result)