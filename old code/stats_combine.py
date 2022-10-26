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
l = [10,20,30,40,50,60,70,80,90,100]
lamda = [ 0.5,1.0,1.5,2]
os.makedirs("resultcd Cd",exist_ok=True)
sum1 = [ ]
for la in lamda:
    sumsum = [ ]
    for t in tags:
        result = []
        for i in range(13):
                # result.append(compute(t,i,[5,10,20,30,50,100]))
            x = np.loadtxt("result_combine/" + t + "_" + str(i) + "_combine" + str(la) + "_lasso_ridge" + ".txt")[:,1]
            sumsum.append(x)

    sumsum = np.array(sumsum)
    sumsum = np.mean(sumsum,axis=0)
    print(sumsum)
            
    # result = np.array(result)
    # np.savetxt("result/" + t +"_fuse_weight_probeless.txt", result)
