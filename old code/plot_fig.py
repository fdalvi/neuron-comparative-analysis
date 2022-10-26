# import numpy as np
# import matplotlib.pyplot as plt
# tags = ["VBG","VBZ","NNPS","DT","TO","CD","JJ"]
# layer =12
# result = dict()
# result['probeless'] = []
# result['selectivity'] = []
# result['lca'] = []
# result['lca_lasso'] = []
# result['lca_ridge'] = []
# result['lca_elasticnet'] = []

# probeless = []
# selectivity = []
# lca_elasticnet = []
# lca_lasso = []
# lca_ridge = []
# lca = []
# for j in range(len(tags)):
#     x = np.loadtxt("result_splits/" + tags[j] + "_" + str(layer) + "_splits.txt")[:,0]
#     probeless_score = x[0::6]
#     selectivity_score = x[1::6]
#     lca_elasticnet_score = x[2::6]
#     lca_lasso_score = x[3::6]
#     lca_ridge_score = x[4::6]
#     lca_score = x[5::6]
#     probeless.append(probeless_score)
#     selectivity.append(selectivity_score)
#     lca_elasticnet.append(lca_elasticnet_score)
#     lca_lasso.append(lca_lasso_score)
#     lca_ridge.append(lca_ridge_score)
#     lca.append(lca_score)
# probeless = np.array(probeless)
# selectivity = np.array(selectivity)
# lca_elasticnet = np.array(lca_elasticnet)
# lca_lasso = np.array(lca_lasso)
# lca_ridge = np.array(lca_ridge)
# lca = np.array(lca)
# probeless, selectivity, lca_elasticnet, lca_lasso, lca_ridge, lca = np.mean(probeless, axis=0), np.mean(selectivity, axis=0),np.mean(lca_elasticnet, axis=0),np.mean(lca_lasso, axis=0),np.mean(lca_ridge, axis=0),np.mean(lca, axis=0)
# x = [10,20,30,40,50,60,70,80,90,100]
# plt.rcParams.update({

#     "xtick.major.size": 5,
#     "xtick.major.pad": 7,
#     "xtick.labelsize": 15,
#     "grid.color": "0.5",
#     "grid.linestyle": "-",
#     "grid.linewidth": 0.5,
#     "lines.linewidth": 2,
#     "lines.color": "g",
# })
# plt.grid(True)
# plt.xticks(np.arange(0,101,10))
# plt.yticks(np.arange(0.75,1.01,0.05))

# plt.plot(x,probeless,"o-",markersize=8,label="probeless: " + str(round(np.mean(probeless),4)))
# plt.plot(x,selectivity,"*-",markersize=8, label = "selectivity: "+ str(round(np.mean(selectivity),4)))
# plt.plot(x,lca_elasticnet,".-",markersize=8,label="lca_elasticnet: "+ str(round(np.mean(lca_elasticnet),4)))
# plt.plot(x,lca_lasso,"+-",markersize=8,label="lca_lasso: "+ str(round(np.mean(lca_lasso),4)))
# plt.plot(x,lca_ridge,'<-',markersize=8,label="lca_ridge: "+ str(round(np.mean(lca_ridge),4)))
# plt.plot(x,lca,'>-',markersize=8,label="lca: "+ str(round(np.mean(lca),4)))
# plt.legend()

# plt.savefig("../figures/layer"+str(layer)+"_result.png")



# import numpy as np
# import matplotlib.pyplot as plt
# tags = ["VBG","VBZ","NNPS","DT","TO","CD","JJ"]
# layer =12
# result = dict()
# result['probeless'] = []
# result['selectivity'] = []
# result['lca'] = []
# result['lca_lasso'] = []
# result['lca_ridge'] = []
# result['lca_elasticnet'] = []

# # probeless = []
# # selectivity = []
# # lca_elasticnet = []
# # lca_lasso = []
# # lca_ridge = []
# # lca = []
# allv = []
# for i in range(13):
#     probeless = []
#     selectivity = []
#     lca_elasticnet = []
#     lca_lasso = []
#     lca_ridge = []
#     lca = []
#     for j in range(len(tags)):
#         x = np.loadtxt("result_splits/" + tags[j] + "_" + str(i) + "_splits.txt")[:,1]
#         probeless_score = x[0::6]
#         selectivity_score = x[1::6]
#         lca_elasticnet_score = x[2::6]
#         lca_lasso_score = x[3::6]
#         lca_ridge_score = x[4::6]
#         lca_score = x[5::6]
#         probeless.append(probeless_score)
#         selectivity.append(selectivity_score)
#         lca_elasticnet.append(lca_elasticnet_score)
#         lca_lasso.append(lca_lasso_score)
#         lca_ridge.append(lca_ridge_score)
#         lca.append(lca_score)
#     probeless = np.array(probeless)
#     selectivity = np.array(selectivity)
#     lca_elasticnet = np.array(lca_elasticnet)
#     lca_lasso = np.array(lca_lasso)
#     lca_ridge = np.array(lca_ridge)
#     lca = np.array(lca)
#     allv.append([np.mean(probeless), np.mean(selectivity),np.mean(lca_elasticnet),np.mean(lca_lasso),np.mean(lca_ridge),np.mean(lca)])
#     # probeless, selectivity, lca_elasticnet, lca_lasso, lca_ridge, lca = np.mean(probeless), np.mean(selectivity),np.mean(lca_elasticnet),np.mean(lca_lasso),np.mean(lca_ridge),np.mean(lca)
# allv = np.array(allv)
# print(allv.shape)
# x = [0,1,2,3,4,5,6,7,8,9,10,11,12]
# plt.rcParams.update({

#     "xtick.major.size": 5,
#     "xtick.major.pad": 7,
#     "xtick.labelsize": 15,
#     "grid.color": "0.5",
#     "grid.linestyle": "-",
#     "grid.linewidth": 0.5,
#     "lines.linewidth": 2,
#     "lines.color": "g",
# })
# plt.grid(True)
# plt.xticks(np.arange(0,13,1))
# plt.yticks(np.arange(0.90,1.01,0.01))

# plt.plot(x,allv[:,0],"o-",markersize=8,label="probeless")
# plt.plot(x,allv[:,1],"*-",markersize=8, label = "selectivity")
# plt.plot(x,allv[:,2],".-",markersize=8,label="lca_elasticnet")
# plt.plot(x,allv[:,3],"+-",markersize=8,label="lca_lasso")
# plt.plot(x,allv[:,4],'<-',markersize=8,label="lca_ridge")
# plt.plot(x,allv[:,5],'>-',markersize=8,label="lca")
# plt.xlabel("Layer",fontsize=18)
# plt.ylabel("Accuracy",fontsize=18)
# plt.legend()
# print(np.mean(allv,axis=0))
# plt.show()
# plt.savefig("../figures/per_layer.png")

import numpy as np
import matplotlib.pyplot as plt
tags = ["VBG","VBZ","NNPS","DT","TO","CD","JJ"]
layer =12
result = dict()
result['probeless'] = []
result['selectivity'] = []
result['lca'] = []
result['lca_lasso'] = []
result['lca_ridge'] = []
result['lca_elasticnet'] = []

# probeless = []
# selectivity = []
# lca_elasticnet = []
# lca_lasso = []
# lca_ridge = []
# lca = []
allv = []
for i in range(13):
    probeless = []
    selectivity = []
    lca_elasticnet = []
    lca_lasso = []
    lca_ridge = []
    lca = []
    a = []
    for j in range(len(tags)):
        x = np.loadtxt("result_splits/" + tags[j] + "_" + str(i) + "_splits.txt")[:,1]
        probeless_score = x[0::6]
        selectivity_score = x[1::6]
        lca_elasticnet_score = x[2::6]
        lca_lasso_score = x[3::6]
        lca_ridge_score = x[4::6]
        lca_score = x[5::6]
        probeless.append(probeless_score)
        selectivity.append(selectivity_score)
        lca_elasticnet.append(lca_elasticnet_score)
        lca_lasso.append(lca_lasso_score)
        lca_ridge.append(lca_ridge_score)
        lca.append(lca_score)
    probeless = np.array(probeless)
    selectivity = np.array(selectivity)
    lca_elasticnet = np.array(lca_elasticnet)
    lca_lasso = np.array(lca_lasso)
    lca_ridge = np.array(lca_ridge)
    lca = np.array(lca)
    a.append(probeless)
    a.append(selectivity)
    a.append(lca_elasticnet)
    a.append(lca_lasso)
    a.append(lca_ridge)
    a.append(lca)
    allv.append(a)
    # allv.append([np.mean(probeless), np.mean(selectivity),np.mean(lca_elasticnet),np.mean(lca_lasso),np.mean(lca_ridge),np.mean(lca)])
    # probeless, selectivity, lca_elasticnet, lca_lasso, lca_ridge, lca = np.mean(probeless), np.mean(selectivity),np.mean(lca_elasticnet),np.mean(lca_lasso),np.mean(lca_ridge),np.mean(lca)
allv = np.array(allv)
allv = np.mean(allv,axis=0)
print(allv.shape)
vbg = allv[:,0,:]
vbz = allv[:,1,:]
nnps = allv[:,2,:]
dt = allv[:,3,:]

x = [10,20,30,40,50,60,70,80,90,100]
plt.rcParams.update({

    "xtick.major.size": 5,
    "xtick.major.pad": 7,
    # "xtick.labelsize": 15,
    "grid.color": "0.5",
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "lines.linewidth": 2,
    "lines.color": "g",
})
plt.grid(True)
plt.xticks(np.arange(10,101,10))
plt.yticks(np.arange(0.70,1.01,0.05))
plt.xlim(8,102)
plt.ylim(0.70,1)

plt.plot(x,vbz[0],"o-",markersize=8,label="probeless")
plt.plot(x,vbz[1] ,"*-",markersize=8, label = "selectivity")
plt.plot(x,vbz[2],".-",markersize=8,label="lca_elasticnet")
plt.plot(x,vbz[3],"+-",markersize=8,label="lca_lasso")
plt.plot(x,vbz[4],'<-',markersize=8,label="lca_ridge")
plt.plot(x,vbz[5],'>-',markersize=8,label="lca")
plt.xlabel("Num. of Neurons",fontsize=15)
plt.ylabel("Accuracy",fontsize=15)
plt.legend()
# print(np.mean(allv,axis=0))
# plt.show()
plt.savefig("../figures/per_vbz.png")