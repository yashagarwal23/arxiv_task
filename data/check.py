# check if all the node values in the node embeddings file are unique

node_embedding_file = open("hepth-th-node-embeddings", "r")
node_embedding_lines = node_embedding_file.readlines()

unique_ids = set()
for line in node_embedding_lines[1:]:
    line_split = line.split()
    node_id = int(line_split[0])
    unique_ids.add(node_id)
    node_emb = [float(x) for x in line_split[1:]]

print(len(unique_ids))

# check the max no of citations
m = 0
with open("train.txt", "r") as file:
    for line in file.readlines():
        m = max(m, len(line.split()))
print(m)    