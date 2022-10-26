import numpy as np 
import sys
import os
sys.path.append("..")

import neurox.interpretation.iou_probe as iou_probe
import neurox.interpretation.linear_probe as linear_probe
import neurox.interpretation.selectivity_probe as selectivity_probe
import neurox.interpretation.ablation as ablation
import argparse
from sklearn.metrics import average_precision_score# transformers_extractor.extract_representations('bert-base-uncased',

def compute(args):
    
    X_train = np.load(args.data_folder + "/train_data_"+ str(args.layer) + "_"+ args.tag + ".npy")
    X_dev = np.load(args.data_folder +"/dev_data_"+ str(args.layer) + "_"+ args.tag + ".npy" )
    y_train = np.load(args.data_folder +"/train_label_" + args.tag + ".npy")
    y_dev = np.load(args.data_folder +"/dev_label_" + args.tag+ ".npy")
    
    ranking = np.loadtxt( args.neuron_folder + "/" + args.setting + "/" + args.tag + "/"+ str(args.layer)+"_neurons.txt", dtype  = np.int32)
    
    number = [10,30,50,70,100]
    result = []
    if args.method == "selectivity":
        score = selectivity_probe.get_neuron_score(X_dev, y_dev)
        for n in number:
            ids = ranking[:n]
            score_n = np.mean(score[ids])
            result.append(score_n)
    elif args.method == "iou":
        _, score = iou_probe.get_neuron_ordering(X_dev, y_dev)
        for n in number:
            ids = ranking[:n]
            score_n = np.mean(score[ids])
            result.append(score_n)
    elif args.method == "lca":
        for n in number:
            selected = ranking[:n]
            X_train_sel = ablation.filter_activations_keep_neurons(X_train, selected)
            X_dev_sel = ablation.filter_activations_keep_neurons(X_dev, selected)
            probe = linear_probe.train_logistic_regression_probe(X_train_sel, y_train, lambda_l1=0.00, lambda_l2=0.00)
            scores_selected = linear_probe.evaluate_probe(probe, X_dev_sel, y_dev)["__OVERALL__"]
            result.append(scores_selected)
    result = np.array(result)
    return result

def main():
    
    parser = argparse.ArgumentParser(
            description="Extract Neurons")
    parser.add_argument('--data_folder', type=str,default='./data',
                       help='folder contains raw data')
    parser.add_argument('--neuron_folder', type=str,default='./neurons',
                       help='folder contains neurons')
    parser.add_argument('--out_path', type=str, default='./metric',
                       help='Output metric path. Default to ./metric/')
                    
    parser.add_argument('--setting', type=str, default="LCA",
                       help='settings for extracting neurons', choices=["random", 'Noreg', 'Gaussian',  'LCA',  'Lasso-01',  'Ridge-01',  'Probeless', 'Selectivity', "IoU"])
    parser.add_argument('--tag', type=str, default="NN",
                       help='choice for tags', choices=["VBG","VBZ","NNPS","DT","TO","CD","JJ", "PRP","MD", "RB", "VBP", "VB", "NNS", "VBN", "POS", "IN", "NN", "CC", "NNP", "VBD"])
    parser.add_argument('--layer', type=int, default=0,
                       help='Choice of layers')
    parser.add_argument('--method', type=str, default="lca",
                       help='Choice of method')
    args = parser.parse_args()   

    
    os.makedirs(args.out_path,exist_ok=True)
    np.savetxt(args.out_path + "/" + args.tag + "_" + str(args.layer)  + args.setting + "_" + args.method +".txt", compute(args))
        
        
if __name__ == "__main__":
    main()

