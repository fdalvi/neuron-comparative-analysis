import numpy as np
import torch
import sys
import os
sys.path.append("..")
import neurox.data.loader as data_loader
import neurox.interpretation.utils as utils
import neurox.interpretation.linear_probe as linear_probe
import neurox.interpretation.probeless as probeless
import neurox.interpretation.gaussian_probe as gaussian_probe
import neurox.interpretation.iou_probe as iou_probe
import neurox.data.extraction.transformers_extractor as transformers_extractor
from imblearn.under_sampling import RandomUnderSampler

import neurox.interpretation.ablation as ablation
from sklearn.metrics.pairwise import cosine_similarity

import argparse

def compute(input_folder, out_path, tag, layer, setting):
   
    X_train = np.load(os.path.join(input_folder, "train_data_"+ str(layer) + "_"+ tag + ".npy"))
    X_dev = np.load(os.path.join(input_folder, "dev_data_"+ str(layer) + "_"+ tag + ".npy") )
    y_train = np.load(os.path.join(input_folder, "train_label_"+ str(layer) + "_"+ tag + ".npy"))
    y_dev = np.load(os.path.join(input_folder, "dev_label_"+ str(layer) + "_"+ tag + ".npy"))
    
    os.makedirs(out_path + "/" + setting  + "/" + tag + "/",exist_ok=True)

    if setting == "LCA":
        probe = linear_probe.train_logistic_regression_probe(X_train, y_train, lambda_l2=0.1,lambda_l1=0.1)
        label2idx = {tag: 1, 'OTHER': 0}
        ranking, _ = linear_probe.get_neuron_ordering(probe, label2idx)
    elif setting == "Noreg":
        probe = linear_probe.train_logistic_regression_probe(X_train, y_train, lambda_l2=0.0,lambda_l1=0.0)
        label2idx = {tag: 1, 'OTHER': 0}
        ranking, _ = linear_probe.get_neuron_ordering(probe, label2idx)
    elif setting == "Lasso-01":
        probe = linear_probe.train_logistic_regression_probe(X_train, y_train, lambda_l2=0.0,lambda_l1=0.1)
        label2idx = {tag: 1, 'OTHER': 0}
        ranking, _ = linear_probe.get_neuron_ordering(probe, label2idx)
    elif setting == "Ridge-01":
        probe = linear_probe.train_logistic_regression_probe(X_train, y_train, lambda_l2=0.1,lambda_l1=0.0)
        label2idx = {tag: 1, 'OTHER': 0}
        ranking, _ = linear_probe.get_neuron_ordering(probe, label2idx)
    elif setting == "Probeless":
        ranking = probeless.get_neuron_ordering(X,y)
    elif setting == "Selectivity":
        mu_plus = np.mean(X_train[y_train==1], axis=0)
        mu_minus = np.mean(X_dev[y_dev==0],axis=0)
        max_activations = np.max(X_train, axis=0)
        min_activations = np.min(X_train, axis=0)

        sel = (mu_plus - mu_minus) / (max_activations - min_activations)
        ranking = np.argsort(np.abs(sel))[::-1]
    elif setting == "IoU":
        ranking = iou_probe.get_neuron_ordering(X_train, y_train)    
    elif setting == "Gaussian":
        probe = gaussian_probe.train_probe(X_train,y_train)
        ranking = gaussian_probe.get_neuron_ordering(probe) 
    elif setting == "random":
        indices = np.arange(768)
        np.random.shuffle(indices) 
        ranking = indices
        
    else:
        print("ERROR input setting")
        exit(0)
    np.savetxt(os.path.join(out_path, setting, tag , str(layer) + "_neurons.txt"),ranking,fmt="%d")

    

def main():
    
    parser = argparse.ArgumentParser(
            description="Extract Neurons")
    parser.add_argument('--input_folder', type=str,default='data',
                       help='folder contains raw data')
    parser.add_argument('--out_path', type=str, default='output',
                       help='Output path. Default to ./output/')
    parser.add_argument('--setting', type=str, default="LCA",
                       help='settings for extracting neurons', choices=["random", 'Noreg', 'Gaussian',  'LCA',  'Lasso-01',  'Ridge-01',  'Probeless', 'Selectivity', "IoU"])
    parser.add_argument('--tag', type=str, default="NN",
                       help='choice for tags', choices=["VBG","VBZ","NNPS","DT","TO","CD","JJ", "PRP","MD", "RB", "VBP", "VB", "NNS", "VBN", "POS", "IN", "NN", "CC", "NNP", "VBD"])
    parser.add_argument('--layer', type=int, default=0,
                       help='Choice of layers')
    args = parser.parse_args()   

    compute(args.input_folder, args.out_path ,args.tag, args.layer, args.setting)
        
if __name__ == "__main__":
    main()