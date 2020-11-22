import numpy as np

node_embeddings = []
node_embedding_file = open("hepth-th-node-embeddings"   , "r")
node_embedding_lines = node_embedding_file.readlines()

idx = 0
node_id_to_idx_map = dict()
for line in node_embedding_lines[1:]:
    line_split = line.split()
    node_id = int(line_split[0])
    if node_id not in node_id_to_idx_map:
        node_id_to_idx_map[node_id] = idx
        idx += 1
    node_emb = [float(x) for x in line_split[1:]]
    node_embeddings.append(node_emb)
node_embeddings.append([0.0 for _ in range(128)])
node_embeddings = np.array(node_embeddings)
print(node_embeddings.shape)

np.save('node_embeddings.npy', node_embeddings)


hep_th_edges = open("hep-th-citations", "r").readlines()
indexed_edge_file = open("hepth-citations-indexed", "w")
adj_list = [[] for _ in range(len(node_id_to_idx_map))]
for edge in hep_th_edges:
    x, y = list(map(int, edge.split()))
    x, y = node_id_to_idx_map[x], node_id_to_idx_map[y]
    indexed_edge_file.write(str(x) + " " + str(y) + "\n")
    adj_list[y].append(x)

print(len(adj_list))
data = [[] for _ in range(len(node_id_to_idx_map))]
for i in range(len(node_id_to_idx_map)):
    data[i].append(str(i))
    for x in adj_list[i]:
        data[i].append(str(x))


data = list(filter(lambda x : len(x) > 5, data))

print(len(data))

train_num = int(0.8*len(data))
valid_num = int(0.1*len(data))
test_num = int(0.1*len(data))

with open("train.txt", "w") as file:
    for i in range(train_num):
        file.write(" ".join(data[i]) + "\n")

with open("valid.txt", "w") as file:
    for i in range(train_num, train_num+valid_num):
        file.write(" ".join(data[i]) + "\n")

with open("test.txt", "w") as file:
    for i in range(train_num+valid_num, train_num+valid_num+test_num):
        file.write(" ".join(data[i]) + "\n")