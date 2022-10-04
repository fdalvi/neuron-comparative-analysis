import numpy as np
tags = ["VBG","VBZ","NNPS","DT","TO","CD","JJ"]
result = []

for i in range(13):
    r = []
    for j in range(len(tags)):
        x = np.loadtxt("result_splits/" + tags[j] + "_" + str(i) + "_lca_l2_01_l1_001.txt")[:,1]
        s = x[::2]
        r.append(s)

    result.append(r)
import pandas as pd
result = np.array(result)
# result = np.mean(result,axis=1)
## convert your array into a dataframe
dfs = []
row = ['0','1','2','3','4','5','6','7','8','9','10','11','12']
col = ['10','20','30','40','50','60','70','80','90','100']
for i in range(len(tags)):
    dfs.append( pd.DataFrame(result[:,i,:], index=row,columns=col))
with pd.ExcelWriter('lca_l2_01_l1_001.xlsx') as writer:  
    for i in range(len(dfs)):
        dfs[i].to_excel(writer, sheet_name=tags[i])
    
## save to xlsx file


# result = np.array(result)
# print(result.shape)
# result = np.mean(result, axis=1)
# for i in range(result.shape[0]):
#     print("----------")
#     for j in range(result.shape[1]):
#         print(result[i][j])    #result.append(np.mean(r))
        #         probeless_score = x[0::6]
#         selectivity_score = x[1::6]
#         lca_elasticnet_score = x[2::6]
#         lca_lasso_score = x[3::6]
#         lca_ridge_score = x[4::6]w    
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
#     probeless, selectivity, lca_elasticnet, lca_lasso, lca_ridge, lca = np.mean(probeless, axis=0), np.mean(selectivity, axis=0),np.mean(lca_elasticnet, axis=0),np.mean(lca_lasso, axis=0),np.mean(lca_ridge, axis=0),np.mean(lca, axis=0)
#     # import pdb;pdb.set_trace()
#     result['probeless'].append(probeless)
#     result['selectivity'].append(selectivity)
#     result['lca_lasso'].append(lca_lasso)
#     result['lca_elasticnet'].append(lca_elasticnet)
#     result['lca_ridge'].append(lca_ridge)
#     result['lca'].append(lca)

# result['probeless'] = np.array(result['probeless'])
# result['selectivity']= np.array(result['selectivity'])
# result['lca_lasso']= np.array(result['lca_lasso'])
# result['lca_elasticnet']= np.array(result['lca_elasticnet'])
# result['lca_ridge']= np.array(result['lca_ridge'])
# result['lca']= np.array(result['lca'])

# np.save("splits",result)