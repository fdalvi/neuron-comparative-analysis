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
from sklearn.metrics import average_precision_score# transformers_extractor.extract_representations('bert-base-uncased',
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
    
    X_train = np.load("data/train_data_"+ str(layer) + "_"+ tag + ".npy")
    X_dev = np.load("data/dev_data_"+ str(layer) + "_"+ tag + ".npy" )
    X_test = np.load("data/test_data_"+ str(layer) + "_"+ tag + ".npy" )
    y_train = np.load("data/train_label_" + tag + ".npy")
    y_test = np.load("data/test_label_" + tag+ ".npy")
    y_dev = np.load("data/dev_label_" + tag+ ".npy")
    # probeless = np.loadtxt("neurons_splits/probeless/"+tag+"/"+str(layer)+"_neurons.txt",dtype = int)
    # selectivity = np.loadtxt("neurons_splits/sel/"+tag+"/"+str(layer)+"_neurons.txt",dtype = int)
    # lca_elasticnet = np.loadtxt("neurons_splits/lca_elasticnet/"+tag+"/"+str(layer)+"_neurons.txt",dtype = int)
    p = []
    score = np.abs(X_train)
    ## X_train.shape = (L,768)
    
    threshold = 0.2
    X_train[np.abs(X_train)< threshold] = 0
    for i in range(768):
        p.append(average_precision_score(y_train,X_train[:,i]))
    p = np.array(p)
    ranking = np.argsort(p)[::-1]
    
    def train_dev_and_eval(X_train, y_train, X_dev, y_dev, X_test, y_test, selected):
        X_train_sel = ablation.filter_activations_keep_neurons(X_train, selected)
        X_dev_sel = ablation.filter_activations_keep_neurons(X_dev, selected)
        X_test_sel =  ablation.filter_activations_keep_neurons(X_test, selected)
        probe = linear_probe.train_logistic_regression_probe(X_train_sel, y_train, lambda_l1=0.00, lambda_l2=0.00)
        scores_selected = linear_probe.evaluate_probe(probe, X_test_sel, y_test)["__OVERALL__"]
        scores_selected_dev = linear_probe.evaluate_probe(probe, X_dev_sel, y_dev)["__OVERALL__"]
        return scores_selected, scores_selected_dev
    result = []
    random_rank = np.arange(768)
    np.random.shuffle(random_rank)
    # print(random_rank)
    # import pdb;pdb.set_trace()
    for n in number:
        # probeless_sel = probeless[:n]
        # probeless_score, probeless_score_dev = train_dev_and_eval(X_train, y_train,X_dev, y_dev, X_test, y_test, probeless_sel)
        # selectivity_sel =selectivity[:n]
        # selectivity_score, selectivity_score_dev = train_dev_and_eval(X_train, y_train,X_dev, y_dev, X_test, y_test, selectivity_sel)
            

        # lca_elasticnet_sel = lca_elasticnet[:n]
        # lca_elasticnet_score, lca_elasticnet_score_dev = train_dev_and_eval(X_train, y_train,X_dev, y_dev, X_test, y_test, lca_elasticnet_sel)
        ids = ranking[:n]
        rdm = random_rank[:n]
        score, score_dev = train_dev_and_eval(X_train, y_train,X_dev, y_dev, X_test, y_test, ids)
        score_r, score_dev_r = train_dev_and_eval(X_train, y_train,X_dev, y_dev, X_test, y_test, rdm)


        result.append([score, score_dev])
        result.append([score_r, score_dev_r])

        # result.append([lca_score, lca_score_dev])
    return result
    # for t in threshold:
    #     print(average_precision_score(X_train, y_train))
    #     X_train[X_train < t] = -100000
    #     X_train[X_train >= t] = 100000
    #     X_train = (X_train + 100000) / 200000
    #     print(X_train)
    #     p
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
l =  [10,20,30,40,50,60,70,80,90,100]
# lamda = [ 0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
os.makedirs("result_splits",exist_ok=True)

for t in tags:
    result = []
    for i in range(13):
        # compute(t,i,l)

            # result.append(compute(t,i,[5,10,20,30,50,100]))
        np.savetxt("result_splits/" + t + "_" + str(i)  + "mask_02_signed.txt", compute(t,i,l))
    # result = np.array(result)
    # np.savetxt("result/" + t +"_fuse_weight_probeless.txt", result)
