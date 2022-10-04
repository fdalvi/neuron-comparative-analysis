import numpy as np
import os
tags = ["PRP","MD", "RB", "VBP", "VB", "NNS", "VBN", "POS", "IN", "NN", "CC", "NNP", "VBD"]
l =  [10,20,30,40,50,60,70,80,90,100]
for t in tags:
    os.makedirs("neurons_splits/random1/"+ t +"/",exist_ok=True)
for t in tags:
    for i in range(13):
        x = np.arange(768)
        np.random.shuffle(x)
        np.savetxt("neurons_splits/random1/"+ t +"/" + str(i)+"_neurons.txt",x, fmt="%d")
