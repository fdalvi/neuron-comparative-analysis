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

def compute(input_folder, out_path, tag,layer, setting):
   
    X_train = np.load("data/train_data_"+ str(layer) + "_"+ tag + ".npy")
    X_dev = np.load("data/dev_data_"+ str(layer) + "_"+ tag + ".npy" )
    y_train = np.load("data/train_label_" + tag + ".npy")
    y_dev = np.load("data/dev_label_" + tag+ ".npy")
    

    
    probe = linear_probe.train_logistic_regression_probe(X_train, y_train, lambda_l2=0.01,lambda_l1=0.01)
    label2idx = {tag: 1, 'OTHER': 0}
    ranking, _ = linear_probe.get_neuron_ordering(probe, label2idx)
    os.makedirs("neurons_splits/lca_l2_001_l1_001_modif/" + tag + "/",exist_ok=True)
    np.savetxt("neurons_splits/lca_l2_001_l1_001_modif/"+tag+"/"+str(layer)+"_neurons.txt",ranking,fmt="%d")
    
    

def main():
    
    parser = argparse.ArgumentParser(
            description="Extract Neurons")
    parser.add_argument('--input_folder', type=str,
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
        
