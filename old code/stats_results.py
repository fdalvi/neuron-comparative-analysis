# tags = ["VBG","VBZ","NNPS","DT","TO","CD","JJ"]
# import numpy as np
# for t in tags:
#     result = np.loadtxt(t+"_corr_lambda025_embed_mean_abs_norm.txt")
#     print(np.mean(result[:,0]-result[:,1]))
tags = ["VBG","VBZ","NNPS","DT","TO","CD","JJ"]
numbers = [ 0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
import numpy as np
re = []
for t in tags:
    r = []
    for i in range(13):
        result = np.loadtxt("result" + "/" + t+ "_" + str(i) + "_10_fuse_weight_probeless_test_dev_fine_grained.txt")
        r.append(result)
        # print(np.mean(result, axis=0))
    r = np.array(r)
    re.append(r)
re = np.array(re)
print(re.shape)

# re = np.mean(re,axis=1)
# re = np.mean(re,axis=0)

for i in range(len(tags)):
    for j in range(13):
        x = re[i,j]
        x1 = x[::2]
        x2 = x[1::2]
        print("Tag: " + str(tags[i]) + " Layer: " + str(j))
        
        T_dev = []
        T_test = []
        for t in range(0,x1.shape[0], 10):
            t1 = np.mean(x1[t:t+10], axis=0)
            t2 = np.mean(x2[t:t+10], axis=0)
            T_dev.append(t1)
            T_test.append(t2)
        T_dev = np.array(T_dev)
        T_test = np.array(T_test)
        dev_max = np.argmax(T_dev[:,2])
        test_max = np.argmax(T_test[:,2])
        print("Lamda test: " + str(numbers[dev_max]))
        print(str(round(T_dev[dev_max][0],4)) + "  " + str(round(T_dev[dev_max][1],4)) + "  " + str(round(T_dev[dev_max][2],4)))
        print("Lamda dev: " + str(numbers[test_max]))
        print(str(round(T_test[test_max][0],4)) + "  " + str(round(T_test[test_max][1],4)) + "  " + str(round(T_test[test_max][2],4)))

# for _ in range(re2.shape[0]):
#     x  = re2[_]
#     with open("result/"+str(_)+"dev.txt","w") as f:
#         for i in range(13):
#             print("layer:"+str(i))
#             f.writelines("layer:"+str(i) +"\n")
#             for l in range(10):
#                 print(str(10 * l + 10) +" "+  str(x[i,l,0]) + " " + str(x[i,l,1]) + " " +str(x[i,l,2]) )
#                 f.writelines(str(10 * l + 10) +" "+  str(x[i,l,0]) + " " + str(x[i,l,1]) + " " +str(x[i,l,2])+"\n")
#             print("AVERAGE" + " " + str(np.mean(x[i,:,0]))  + " " + str(np.mean(x[i,:,1])) + " " +str(np.mean(x[i,:,2])))
#             f.writelines("AVERAGE" + " " + str(np.mean(x[i,:,0]))  + " " + str(np.mean(x[i,:,1])) + " " +str(np.mean(x[i,:,2]))+"\n")
    # r = r.reshape(-1,3)
    # print(np.mean(r
    # print(np.mean(r[:,2]-r[:,1]))