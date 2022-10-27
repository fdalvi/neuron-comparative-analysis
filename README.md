# Neuron Comparative Analysis

This is the official repo of our paper entitled ...

## Requirements

We don't have any additional requirements besides the requirements of NeuroX https://github.com/fdalvi/NeuroX

## Usage

### Step 1: preprocess the dataset

```bash
python preprocess_dataset.py --input_folder . --out_path data --words word.txt --labels label.txt --model_name_or_path bert-base-uncased
```

Customize your dataset to replace word.txt and label.txt. You data should satisfy the following format

word.txt

```
Rockwell International Corp. 's Tulsa unit said it signed a tentative agreement extending its contract with Boeing Co. to provide structural parts for Boeing 's 747 jetliners .
Rockwell said the agreement calls for it to supply 200 additional so-called shipsets for the planes .
These include , among other parts , each jetliner 's two major bulkheads , a pressure floor , torque box , fixed leading edges for the wings and an aft keel beam .
Under the existing contract , Rockwell said , it has already delivered 793 of the shipsets to Boeing .
Rockwell , based in El Segundo , Calif. , is an aerospace , electronics , automotive and graphics concern .
```

label.txt

```
NNP NNP NNP POS NNP NN VBD PRP VBD DT JJ NN VBG PRP$ NN IN NNP NNP TO VB JJ NNS IN NNP POS CD NNS .
NNP VBD DT NN VBZ IN PRP TO VB CD JJ JJ NNS IN DT NNS .
DT VBP , IN JJ NNS , DT NN POS CD JJ NNS , DT NN NN , NN NN , VBN VBG NNS IN DT NNS CC DT JJ NN NN .
IN DT VBG NN , NNP VBD , PRP VBZ RB VBN CD IN DT NNS TO NNP .
NNP , VBN IN NNP NNP , NNP , VBZ DT NN , NNS , JJ CC NNS NN .
```

This step will output the activations of your dataset, and split the activations dataset into train/dev/test set per tag. 

### Step 2: Extract Neurons

```bash
tags="VBG VBZ NNPS DT TO CD JJ PRP MD RB VBP VB NNS VBN POS IN NN CC NNP VBD"
settings="random Noreg Gaussian LCA Lasso-01 Ridge-01 Probeless Selectivity IoU"
layers="0 1 2 3 4 5 6 7 8 9 10 11 12"
for k in $layers;
do 
    for i in $tags;
    do
        for j in $settings;
        do  
        python extract_neurons.py --input_folder data --out_path neurons --setting $j --tag $i --layer $k ;  
        done
    done
done
```

This step extracts neurons for each method we discuss. You can customize the settings. Please note that running Gaussian method is slow.

### Step 3: Compute Metrics

```bash
tags="VBG VBZ NNPS DT TO CD JJ PRP MD RB VBP VB NNS VBN POS IN NN CC NNP VBD"
settings="random Noreg Gaussian LCA Lasso-01 Ridge-01 Probeless Selectivity IoU"
layers="0 1 2 3 4 5 6 7 8 9 10 11 12"
methods="lca selectivity iou"
for m in $methods;
do
    for k in $layers;
    do 
        for i in $tags;
        do
            for j in $settings;
            do  
            python compute_metric.py --data_folder data --neuron_folder neurons --out_path metrics --setting $j --tag $i --layer $k --method $m ;  
            done
        done
    done
done
```

This step computes three metrics for evaluation, namely classification accuracy (lca), selectivity and IoU based score

### Step 4: Compatiability score (AvgOverlap and Neuron Vote)

```bash
python avg_overlap.py --input_folder neurons --out_path score --setting Gaussian,LCA,Lasso-01,Ridge-01,Probeless,Selectivity,IoU --baseline_methods Gaussian,LCA,Lasso-01,Ridge-01 --tags NNPS,NN,VBN --layers 0,1,2,3 --num_of_neurons 10,20,30,40,100
python neuron_vote.py --input_folder neurons --out_path score --setting Gaussian,LCA,Lasso-01,Ridge-01,Probeless,Selectivity,IoU --baseline_methods Gaussian,LCA,Lasso-01,Ridge-01 --tags NNPS,NN,VBN --layers 0,1,2,3 --num_of_neurons 10,20,30,40,100

```

This step computes the compatibility score. The setting is our target method, the baseline methods are those methods for comparison against. Suppose you have a new method A and want to compare it against B,C,D, you will write the command --setting A and --baseline methods B,C,D.

The output is a matrix with shape (num_of_neurons, layers, tags, baseline methods). You can analyze your method at different dimensions.

## How to Contribute 

If you have proposed a new neuron interpretation method, it is recommended to first try to integrate it into neurox, and then you can add it into this framework to evaluate. If there are any comments or problems, feel free to open an issue and ask us.

