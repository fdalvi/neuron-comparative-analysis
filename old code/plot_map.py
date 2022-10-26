import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import matplotlib

settings = ['gaussian_probe', 'lca', 'lca_l2_001_l1_001', 'lca_l2_001_l1_01', 'lca_l2_01_l1_001', 'lca_l2_01_l1_01', 'lca_lasso_001', 'lca_lasso_01',  'lca_ridge_001', 'lca_ridge_01',  'probeless', 'sel']

for i in range(10,101,10):
    h = []
    cnt = 0
    with open("overlap_res1.txt") as f:
        for line in f:
            if line == str(i) + "\n":
                cnt = 1
            if cnt == 1:
                h.append(line)
            if line == str(i+10) + "\n":
                cnt = 0
    h = h[1:-1]
    m = np.zeros((12,12))
    for l in h:
        l = l.split(" ")
        if l[0] in settings and l[2] in settings:
            id1 = settings.index(l[0])
            id2 = settings.index(l[2])
            m[id1][id2] = float(l[4])
            m[id2][id1] = float(l[4])
    fig, ax = plt.subplots()
    im = ax.imshow(m)
    ax.set_xticks(np.arange(len(settings)), labels=settings)
    ax.set_yticks(np.arange(len(settings)), labels=settings)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    for p in range(len(settings)):
        for q in range(len(settings)):
            text = ax.text(q, p, round(m[p][q],2),
                        ha="center", va="center", color="w",size=8)

    ax.set_title("Heatmap " + str(i))
    fig.tight_layout()
    plt.legend()
    fig.savefig("heat" + str(i) + ".png")
    plt.cla()



# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt

# vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
#               "potato", "wheat", "barley"]
# farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
#            "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

# harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
#                     [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
#                     [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
#                     [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
#                     [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
#                     [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
#                     [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])


# fig, ax = plt.subplots()
# im = ax.imshow(harvest)

# # Show all ticks and label them with the respective list entries
# ax.set_xticks(np.arange(len(farmers)), labels=farmers)
# ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)

# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")

# # Loop over data dimensions and create text annotations.
# for i in range(len(vegetables)):
#     for j in range(len(farmers)):
#         text = ax.text(j, i, harvest[i, j],
#                        ha="center", va="center", color="w")

# ax.set_title("Harvest of local farmers (in tons/year)")
# fig.tight_layout()
# plt.show()
