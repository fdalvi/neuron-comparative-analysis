import os
import numpy as np
import torch
import sys
sys.path.append("..")
import neurox.data.loader as data_loader
import neurox.data.extraction.transformers_extractor as transformers_extractor
from imblearn.under_sampling import RandomUnderSampler
import os
import neurox.interpretation.utils as utils
import argparse
import neurox.interpretation.ablation as ablation
os.system("unzip sample_data.zip")
parser = argparse.ArgumentParser(
            description="Extract Data")
parser.add_argument('--input_folder', type=str,default='.',
                       help='folder contains raw data')
parser.add_argument('--out_path', type=str, default='data',
                       help='Output path. Default to ./output/')

args = parser.parse_args()   

os.makedirs(args.out_path, exist_ok=True)
# transformers_extractor.extract_representations('bert-base-uncased',
#     args.input_folder + '/word.txt',
#     args.input_folder + '/activations.json',
#     device="cpu",
#     aggregation="average" #last, first
# )



def extract(tag):
    X, y, mapping = utils.create_tensors(tokens, activations, 'NN',binarized_tag=tag)
    label2idx, idx2label, src2idx, idx2src = mapping 
    rus = RandomUnderSampler(random_state=0)
    X,y = rus.fit_resample(X, y)
    X = X.reshape(-1,13,768)
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
    dev_ratio = 0.15
    test_ratio = 0.15
    X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(X,y, dev_ratio, test_ratio)

    for i in range(13):
        np.save(args.out_path + "/train_data_"+ str(i) + "_"+ tag, X_train[:,i,:])
        np.save(args.out_path + "/test_data_"+ str(i) + "_"+ tag, X_test[:,i,:])

        np.save(args.out_path + "/dev_data_"+ str(i) + "_"+ tag, X_dev[:,i,:])

    np.save(args.out_path + "/train_label_"+ tag, y_train)

    np.save(args.out_path + "/test_label_"+  tag, y_test)

    np.save(args.out_path + "/dev_label_"+  tag, y_dev)


activations, num_layers = data_loader.load_activations( args.input_folder + '/activations.json', 768)
tokens = data_loader.load_data(args.input_folder + '/word.txt', args.input_folder + '/label.txt', activations, 512 )

tags = ["VBG","VBZ","NNPS","DT","TO","CD","JJ", "PRP","MD", "RB", "VBP", "VB", "NNS", "VBN", "POS", "IN", "NN", "CC", "NNP", "VBD"]
for t in tags:
    extract(t)

