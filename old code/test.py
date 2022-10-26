# import os
# h = os.listdir(".")
# print(h)
# s = 0
# c = 0
# for i in h:
#     if "20" in i and "label" in i:
#         with open(i) as f:
#             for line in f:
#                 t = line.split(" ")
#                 s+= len(t)
#                 c+=1
# print(s)
# print(c)
with open("split_col.txt", encoding="utf-8") as f:
    for line in f:
        line = line[0:-7]
        print(line)