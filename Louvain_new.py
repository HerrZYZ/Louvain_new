import pandas as pd
import numpy as np
from sknetwork.clustering import Louvain, modularity
from scipy.sparse import csr_matrix


df_nodeedge = pd.read_csv('undirected_training_graph_numeric.csv')

# print (df_nodeedge)
louvain = Louvain( resolution=-2, modularity='newman', n_aggregations=-1, shuffle_nodes= True, verbose=True,return_aggregate= True)

df_from=df_nodeedge['from_address']
df_to=df_nodeedge['to_address']
weights=df_nodeedge['size']


row = df_from.to_numpy()
col = df_to.to_numpy()
data = weights.to_numpy()

# print (row)
adjacency = csr_matrix((data, (row, col)))

labels = louvain.fit_transform(adjacency)

print (modularity(adjacency, labels))

labels_unique, counts = np.unique(labels, return_counts=True)
print(labels_unique, counts)


print (counts[100])


f = open("Louvain_results_reso1_addsmall.txt", "w")
f.write('modularity')
f.write(str(modularity(adjacency, labels)))

f.write('counts')

for i in range(len(counts)):
    f.write(str(counts[i]))
    f.write(' ')

f.close()