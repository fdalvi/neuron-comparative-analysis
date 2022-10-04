import numpy as np
import torch
import sys
import os
sys.path.append("..")
import neurox.data.loader as data_loader
import neurox.interpretation.utils as utils
import neurox.interpretation.linear_probe as linear_probe
import neurox.interpretation.probeless as probeless
import neurox.data.extraction.transformers_extractor as transformers_extractor
from imblearn.under_sampling import RandomUnderSampler

import neurox.interpretation.ablation as ablation
from sklearn.metrics.pairwise import cosine_similarity

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

def compute(tag,layer):
    X, y, mapping = utils.create_tensors(tokens, activations, 'NN',binarized_tag=tag)
    X = X.reshape(-1,13,768)[:,layer,:]
    label2idx, idx2label, src2idx, idx2src = mapping 
    rus = RandomUnderSampler(random_state=0)
    X,y = rus.fit_resample(X, y)
    probe = linear_probe.train_logistic_regression_probe(X, y, lambda_l1=0.000001, lambda_l2=0.000001)
    we = probe.linear.weight
    we = np.sqrt(np.sum(we.detach().numpy()**2,axis=0))
    we_ranking = np.argsort(we)[::-1]
    os.makedirs("neurons/weight/",exist_ok=True)
    os.makedirs("neurons/weight/" + tag + "/",exist_ok=True)

    np.savetxt("neurons/weight/"+tag+"/"+str(layer)+"_neurons.txt",we_ranking,fmt="%d")
    np.savetxt("neurons/weight/"+tag+"/"+str(layer)+"_values.txt",we)
    # print(we_ranking)
    # _, probeless_neurons,values = probeless.get_neuron_ordering(X,y)
    # os.makedirs("neurons/probeless/",exist_ok=True)
    # os.makedirs("neurons/probeless/" + tag + "/",exist_ok=True)

    # np.savetxt("neurons/probeless/"+tag+"/"+str(layer)+"_neurons.txt",probeless_neurons,fmt="%d")
    # np.savetxt("neurons/probeless/"+tag+"/"+str(layer)+"_values.txt",values)

    # sim = cosine_similarity(X.T)
    # sim = np.abs(sim)
    # embeddings, ranking, values = probeless.get_neuron_ordering(X,y)
    # selected = [ranking[0]]
    # lamba = 0.25
    # scores = values.copy()
    # embeddings[0] = (embeddings[0] - np.min(embeddings[0])) / (np.max(embeddings[0]) - np.min(embeddings[0]))
    # embeddings[1] = (embeddings[1] - np.min(embeddings[1])) / (np.max(embeddings[1]) - np.min(embeddings[1]))
    # X_train_sel = ablation.filter_activations_keep_neurons(X_train, selected)
    # X_test_sel =  ablation.filter_activations_keep_neurons(X_test, selected)
    
    # scores = scores / (np.max(scores))
    # for i in range(number - 1):
    #     #scores -= lamba * np.mean(sim[selected], axis=1)
    #     temp0 = embeddings[0] - lamba * np.mean(sim[selected]) * np.mean(embeddings[1][selected])
    #     temp1 = embeddings[1] - lamba * np.mean(sim[selected]) * np.mean(embeddings[0][selected])
    #     order = np.argsort(np.abs(embeddings[0]-embeddings[1]))[::-1]
    #     # embeddings[0] -= lamba * sim[selected[i]] * np.sign(embeddings[0][selected[i]])
    #     # embeddings[1] -= lamba * sim[selected[i]] * np.sign(embeddings[1][selected[i]])
    #     #order = np.argsort((scores - lamba * np.mean(sim[selected],axis=0)))[::-1]
    #     for j in range(len(order)):
    #         if order[j] not in selected:
    #             selected.append(order[j])
    #             break
    #     # lamba = lamba / 2
    #     # selected.append(order[0])
    # rus = RandomUnderSampler(random_state=0)
    
    # X,y = rus.fit_resample(X, y)
    # probe = linear_probe.train_logistic_regression_probe(X, y, lambda_l1=0.00, lambda_l2=0.00)
    # we = probe.linear.weight
    # we = np.sum(np.abs(we.detach().numpy()),axis=0)
    # we_ranking = np.argsort(we)[::-1]
    # print(we_ranking)
    # import pdb;pdb.set_trace()
    # def train_test_split(X,y,ratio):
    #     index = np.arange(len(y))
    #     np.random.shuffle(index)
    #     y = y[index]
    #     X = X[index]
    #     train_len = int(ratio * len(y))
    #     return X[:train_len], y[:train_len], X[train_len:], y[train_len:]
    # train_test_split_ratio = 0.7
    # X_train, y_train, X_test, y_test = train_test_split(X,y, train_test_split_ratio)

    # X_train_sel = ablation.filter_activations_keep_neurons(X_train, selected)
    # X_test_sel =  ablation.filter_activations_keep_neurons(X_test, selected)
    # probe = linear_probe.train_logistic_regression_probe(X_train_sel, y_train, lambda_l1=0.00, lambda_l2=0.00)
    # scores_selected = linear_probe.evaluate_probe(probe, X_test_sel, y_test, idx_to_class=idx2label)["__OVERALL__"]

    # X_train_probeless = ablation.filter_activations_keep_neurons(X_train, probeless_neurons)
    # X_test_probeless =  ablation.filter_activations_keep_neurons(X_test, probeless_neurons)
    # probe_baseline = linear_probe.train_logistic_regression_probe(X_train_probeless, y_train, lambda_l1=0.00, lambda_l2=0.00)
    # scores_probeless = linear_probe.evaluate_probe(probe_baseline, X_test_probeless, y_test, idx_to_class=idx2label)["__OVERALL__"]
    # return [scores_selected,scores_probeless]

activations, num_layers = data_loader.load_activations('../activations_pos.json', 768)
tokens = data_loader.load_data('../Data_for_Yimin/Data_for_Yimin/POS/wsj.20.test.conllx.word',
                                '../Data_for_Yimin/Data_for_Yimin/POS/wsj.20.test.conllx.label',
                                activations,
                                512 # max_sent_l
                                )
# activations, num_layers = data_loader.load_activations('activations_sample_pos.json', 768)
# tokens = data_loader.load_data('../Data_for_Yimin/Data_for_Yimin/POS/sample.word.txt',
#                                 '../Data_for_Yimin/Data_for_Yimin/POS/sample.label.txt',
#                                 activations,
#                                 512 # max_sent_l
#                                 )

tags = ["VBG","VBZ","NNPS","DT","TO","CD","JJ"]
for t in tags:
    # result = []
    for i in range(13):
        compute(t,i)
        # result.append(compute(t,i,2))
        # result.append(compute(t,i,3))
        # result.append(compute(t,i,4))
        # result.append(compute(t,i,5))
        # result.append(compute(t,i,8))
        # result.append(compute(t,i,10))
        # result.append(compute(t,i,20))
        # result.append(compute(t,i,30))
        # result.append(compute(t,i,50))
        # result.append(compute(t,i,100))
    # result = np.array(result)
    # np.savetxt(t+"_corr_lambda025_embed_mean_abs_norm.txt", result)
