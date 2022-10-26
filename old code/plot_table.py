import numpy as np
tags = ["VBG","VBZ","NNPS","DT","TO","CD","JJ"]
result = dict()
result['probeless'] = []
result['selectivity'] = []
result['lca'] = []
result['lca_lasso'] = []
result['lca_ridge'] = []
result['lca_elasticnet'] = []

for i in range(13):
    probeless = []
    selectivity = []
    lca_elasticnet = []
    lca_lasso = []
    lca_ridge = []
    lca = []
    for j in range(len(tags)):
        x = np.loadtxt("result_splits/" + tags[j] + "_" + str(i) + "_splits.txt")[:,0]
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
    probeless, selectivity, lca_elasticnet, lca_lasso, lca_ridge, lca = np.mean(probeless, axis=0), np.mean(selectivity, axis=0),np.mean(lca_elasticnet, axis=0),np.mean(lca_lasso, axis=0),np.mean(lca_ridge, axis=0),np.mean(lca, axis=0)
    # import pdb;pdb.set_trace()
    result['probeless'].append(probeless)
    result['selectivity'].append(selectivity)
    result['lca_lasso'].append(lca_lasso)
    result['lca_elasticnet'].append(lca_elasticnet)
    result['lca_ridge'].append(lca_ridge)
    result['lca'].append(lca)

result['probeless'] = np.array(result['probeless'])
result['selectivity']= np.array(result['selectivity'])
result['lca_lasso']= np.array(result['lca_lasso'])
result['lca_elasticnet']= np.array(result['lca_elasticnet'])
result['lca_ridge']= np.array(result['lca_ridge'])
result['lca']= np.array(result['lca'])

print(np.mean(result['probeless'],axis=0))
print(np.mean(result['selectivity'],axis=0))
print(np.mean(result['lca_lasso'],axis=0))
print(np.mean(result['lca_elasticnet'],axis=0))
print(np.mean(result['lca_ridge'],axis=0))
print(np.mean(result['lca'],axis=0))