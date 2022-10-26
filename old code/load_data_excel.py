import pandas as pd
settings = ['gaussian', 'lca', 'lca_l2_001_l1_001', 'lca_l2_001_l1_01', 'lca_l2_01_l1_001', 'lca_l2_01_l1_01', 'lasso_001', 'lasso_01',  'ridge_001', 'ridge_01',  'probeless', 'selectivity', "mask"]
tags = ["VBG","VBZ","NNPS","DT","TO","CD","JJ"]
z = []
for i in settings:
    l = []
    for j in tags:
        x = pd.read_excel("tables/" + i + ".xlsx",sheet_name=j)
        x = x.to_numpy()[:,1:,]
        l.append(x)
    z.append(l)

import numpy as np
z = np.array(z)
print(z.shape)

smat = []
for t in tags:
    tmat = []
    for i in range(13):
        x = np.loadtxt("result_splits/" + t + "_" + str(i)  +  "_random_1.txt")[:,1]
        tmat.append(x)
    smat.append(tmat)

r1 = np.array(smat)
print(r1.shape)
r1 = r1[np.newaxis,:]
import pdb;pdb.set_trace()
z = np.concatenate([z,r1])
np.save("all_results.npy",z)
