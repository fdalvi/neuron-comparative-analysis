python preprocess_dataset.py --input_folder . --out_path data
python extract_neurons.py --input_folder data --out_path neurons --setting LCA --tag NN --layer 1
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
python avg_overlap.py --input_folder neurons --out_path score --setting Gaussian,LCA,Lasso-01,Ridge-01,Probeless,Selectivity,IoU --baseline_methods Gaussian,LCA,Lasso-01,Ridge-01 --tags NNPS,NN,VBN --layers 0,1,2,3 --num_of_neurons 10,20,30,40,100
python neuron_vote.py --input_folder neurons --out_path score --setting Gaussian,LCA,Lasso-01,Ridge-01,Probeless,Selectivity,IoU --baseline_methods Gaussian,LCA,Lasso-01,Ridge-01 --tags NNPS,NN,VBN --layers 0,1,2,3 --num_of_neurons 10,20,30,40,100
