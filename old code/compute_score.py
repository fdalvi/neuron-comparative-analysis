import numpy as np 
import sys
import os
sys.path.append("..")

import neurox.interpretation.iou_probe as iou_probe
from sklearn.metrics import average_precision_score# transformers_extractor.extract_representations('bert-base-uncased',

def compute(tag, layer, number, method):
    
    X_train = np.load("data/train_data_"+ str(layer) + "_"+ tag + ".npy")
    X_dev = np.load("data/dev_data_"+ str(layer) + "_"+ tag + ".npy" )
    X_test = np.load("data/test_data_"+ str(layer) + "_"+ tag + ".npy" )
    y_train = np.load("data/train_label_" + tag + ".npy")
    y_test = np.load("data/test_label_" + tag+ ".npy")
    y_dev = np.load("data/dev_label_" + tag+ ".npy")
    
    ranking = np.loadtxt("neurons_splits/" + method + "/" + tag + "/"+ str(layer)+"_neurons.txt", dtype  = np.int32)
    # p = []
    # score = np.abs(X_train)
    
    # threshold = 0.05
    # X_dev[np.abs(X_dev)< threshold] = 0
    # for i in range(768):
    #     p.append(average_precision_score(y_dev,X_dev[:,i]))
    # p = np.array(p)    
    mu_plus = np.mean(X_dev[y_dev==1], axis=0)
    mu_minus = np.mean(X_dev[y_dev==0],axis=0)
    max_activations = np.max(X_dev, axis=0)
    min_activations = np.min(X_dev, axis=0)
 
    sel = (mu_plus - mu_minus) / (max_activations - min_activations)
    p = np.abs(sel)
    result = []
    
    for n in number:
        ids = ranking[:n]
        score = np.mean(p[ids])
        result.append(score)

    return result



tags = ["VBG","VBZ","NNPS","DT","TO","CD","JJ", "PRP","MD", "RB", "VBP", "VB", "NNS", "VBN", "POS", "IN", "NN", "CC", "NNP", "VBD"]
l =  [10,20,30,40,50,60,70,80,90,100]
# lamda = [ 0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
os.makedirs("result_splits",exist_ok=True)
# settings = ["random1","random2", "random3"]
settings = ['lca_l2_001_modif']
for s in settings:
    for t in tags:
        result = []
        for i in range(13):
            np.savetxt("result_splits/" + t + "_" + str(i)  +  s + "_sel.txt", compute(t,i,l,s))
    
