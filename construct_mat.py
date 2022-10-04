import numpy as np
tags = ["VBG","VBZ","NNPS","DT","TO","CD","JJ", "PRP","MD", "RB", "VBP", "VB", "NNS", "VBN", "POS", "IN", "NN", "CC", "NNP", "VBD"]
l =  [10,20,30,40,50,60,70,80,90,100]
settings = [ 'lca_l2_0_l1_0_modif', 'lca_l2_001_l1_01_modif', 'lca_l1_001_modif',   'lca_l1_01_modif','lca_l2_001_modif', 'lca_l2_01_modif']
r = []
for s in settings:
    smat = []
    for t in tags:
        tmat = []
        for i in range(13):
            x = np.loadtxt("result_splits/" + t + "_" + str(i)  + "_" + s + ".txt")[:,1]
            tmat.append(x)
        smat.append(tmat)
    r.append(smat)
r1 = np.array(r)
r1 = np.mean(r1, axis=1)
r1 = np.mean(r1, axis=1)

print(r1)
# r1[10] = np.mean(r1[10:13])
for i in range(6):
    for j in range(10):
        print(r1[i][j])
    print("\n")
import pdb;pdb.set_trace()

# np.save("iou_mat_new.npy", r1[0:11])
# import numpy as np
# x = np.load("iou_mat_new.npy")
# x = np.mean(x, axis=1)
# x = np.mean(x, axis=1)
# print(x.shape)
# for i in range(x.shape[0]):
#     for j in range(x.shape[1]):
#         print(x[i][j])
#     print("\n")