import xlwt
import numpy as np
workbook = xlwt.Workbook(encoding='utf-8')
tags = ['VBG','VBZ','NNPS','DT','CD','TO','JJ']
numbers = [5,10,20,30,50,100]
result_dict = {}
ordering = {}
for i in range(len(tags)):
    result_dict[tags[i]] = np.loadtxt(tags[i]+"_corr_lambda05.txt")
    # ordering[tags[i]] = np.loadtxt(tags[i] + "_result_ordering.txt")
print(result_dict)
for i in range(13):   
    sheet1 = workbook.add_sheet("Layer " + str(i))
    sheet1.write(0,0,"results")
    for j in range(len(numbers)):
        sheet1.write(1, j+1, str(numbers[j]))
    idx = 1
    for t in tags:
        sheet1.write(idx, 0, t)
        sheet1.write(idx + 1, 0, "Probeless")
        sheet1.write(idx + 2, 0, "Weight")
        sheet1.write(idx+ 3, 0 " Fuse")
        # sheet1.write(idx + 3, 0, "Random Neurons")
        # sheet1.write(idx + 4, 0 ,"Ordering Neurons")
        res = result_dict[t][i*6:i*6+6]
        # orde = ordering[t][i*7:i*7+7].reshape(7,1)
        # print(res.shape)
        # print(orde.shape)
        # res = np.concatenate([res,orde],axis=1)
        for col in range(2):
            for line in range(6):
                sheet1.write(idx + col + 1, line + 1, str(round(res[line][col],3)))
        idx += 4
workbook.save("corr_lambda05.xls")


    
