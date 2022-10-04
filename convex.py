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

def compute(tag, layer, number, lamda):
    X, y, mapping = utils.create_tensors(tokens, activations, 'NN',binarized_tag=tag)
    X = X.reshape(-1,13,768)[:,layer,:]
    label2idx, idx2label, src2idx, idx2src = mapping
    # _, probeless_neurons,_ = probeless.get_neuron_ordering(X,y)
    # probeless_neurons = probeless_neurons[:number]
    probeless_neurons = np.loadtxt("neurons/probeless/" + tag + "/" + str(layer) + "_neurons.txt",dtype = int)
    probeless_values = np.loadtxt("neurons/probeless/" + tag + "/" + str(layer) + "_values.txt")
    weighted_neurons = np.loadtxt("neurons/weight/" + tag + "/" + str(layer) + "_neurons.txt",dtype=int)
    weighted_values = np.loadtxt("neurons/weight/" + tag + "/" + str(layer) + "_values.txt")
    
    rus = RandomUnderSampler(random_state=0)
    X,y = rus.fit_resample(X, y)
    # probeless_values = np.exp(probeless_values)
    def train_dev_test_split(X,y,dev_ratio, test_ratio):
        index = np.arange(len(y))
        np.random.shuffle(index)
        y = y[index]
        X = X[index]
        ratio = 1 - dev_ratio - test_ratio
        train_len = int(ratio * len(y))
        test_len = int(test_ratio * len(y))
        dev_len = int(dev_ratio * len(y))
        train_len = len(y) - test_len - dev_len
        return X[:train_len], y[:train_len], X[train_len:train_len + dev_len], y[train_len:train_len + dev_len], X[train_len + dev_len: ], y[train_len + dev_len: ]
    # train_test_split_ratio = 0.7
    dev_ratio = 0.15
    test_ratio = 0.15
    X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(X,y, dev_ratio, test_ratio)
    def train_dev_and_eval(X_train, y_train, X_dev, y_dev, X_test, y_test, selected):
        X_train_sel = ablation.filter_activations_keep_neurons(X_train, selected)
        X_dev_sel = ablation.filter_activations_keep_neurons(X_dev, selected)
        X_test_sel =  ablation.filter_activations_keep_neurons(X_test, selected)
        probe = linear_probe.train_logistic_regression_probe(X_train_sel, y_train, lambda_l1=0.00, lambda_l2=0.00)
        scores_selected = linear_probe.evaluate_probe(probe, X_test_sel, y_test, idx_to_class=idx2label)["__OVERALL__"]
        scores_selected_dev = linear_probe.evaluate_probe(probe, X_dev_sel, y_dev, idx_to_class=idx2label)["__OVERALL__"]
        return scores_selected, scores_selected_dev

    probeless_values = (probeless_values - np.min(probeless_values)) / (np.max(probeless_values) - np.min(probeless_values))
    weighted_values = (weighted_values - np.min(weighted_values)) / (np.max(weighted_values) - np.min(weighted_values))
    fuse_values = np.stack([probeless_values,weighted_values],axis=1)
    hull = ConvexHull(np.stack([probeless_values,weighted_values],axis=1))
    print(hull.simplices)

    import pdb;pdb.set_trace()
    result = []
    for l in lamda:
        fuse_values = probeless_values + weighted_values * l
        fuse_neurons = np.argsort(fuse_values)[::-1]
        for n in number:
            probeless_sel = probeless_neurons[:n]
            probeless_score, probeless_score_dev = train_dev_and_eval(X_train, y_train,X_dev, y_dev, X_test, y_test, probeless_sel)
            weighted_sel = weighted_neurons[:n]
            weighted_score, weighted_score_dev = train_dev_and_eval(X_train, y_train,X_dev, y_dev, X_test, y_test, weighted_sel)
            fuse_sel = fuse_neurons[:n]
            fuse_score, fuse_score_dev = train_dev_and_eval(X_train, y_train, X_dev, y_dev, X_test, y_test, fuse_sel)
            result.append([probeless_score, weighted_score, fuse_score])
            result.append([probeless_score_dev, weighted_score_dev, fuse_score_dev])
    return result

# activations, num_layers = data_loader.load_activations('../activations_pos.json', 768)
# tokens = data_loader.load_data('../Data_for_Yimin/Data_for_Yimin/POS/wsj.20.test.conllx.word',
#                                 '../Data_for_Yimin/Data_for_Yimin/POS/wsj.20.test.conllx.label',
#                                 activations,
#                                 512 # max_sent_l
#                                 )
activations, num_layers = data_loader.load_activations('activations_sample_pos.json', 768)
tokens = data_loader.load_data('../Data_for_Yimin/Data_for_Yimin/POS/sample.word.txt',
                                '../Data_for_Yimin/Data_for_Yimin/POS/sample.label.txt',
                                activations,
                                512 # max_sent_l
                                )

tags = ["VBG","VBZ","NNPS","DT","TO","CD","JJ"]
l = [10,20,30,40,50,60,70,80,90,100]
lamda = [ 0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
os.makedirs("resultcd Cd",exist_ok=True)

for t in tags:
    result = []
    for i in range(13):
            # result.append(compute(t,i,[5,10,20,30,50,100]))
        np.savetxt("result/" + t + "_" + str(i) + "_101_fuse_weight_probeless_test_dev_fine_grained" + ".txt", compute(t,i,l, lamda))
    # result = np.array(result)
    # np.savetxt("result/" + t +"_fuse_weight_probeless.txt", result)
