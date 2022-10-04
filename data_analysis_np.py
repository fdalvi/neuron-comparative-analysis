import numpy as np
tags = ["VBG","VBZ","NNPS","DT","TO","CD","JJ"]
l =  [10,20,30,40,50,60,70,80,90,100]
settings = ['gaussian_probe', 'lca', 'lca_l2_001_l1_001', 'lca_l2_001_l1_01', 'lca_l2_01_l1_001', 'lca_l2_01_l1_01', 'lca_lasso_001', 'lca_lasso_01',  'lca_ridge_001', 'lca_ridge_01',  'probeless', 'sel', "iou_probe"]

x = np.load("sel_mat.npy") ### (13,7,13,10)
# 13: settings
# 7: tags
# 13: layers
# 10: num of neurons
print(x)
me = np.mean(x, axis=1)
me = np.mean(me,axis=1)
me = np.mean(me,axis=1)
print(me)