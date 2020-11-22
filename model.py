import torch
from torch import embedding
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, node_embeddings, padding_idx, emb_size, hidden_size, out_size, num_layers = 3, dropout = 0.3, train_embeds = False):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(node_embeddings.size(0), node_embeddings.size(1), padding_idx=padding_idx)
        self.embedding.weight.data.copy_(node_embeddings)
        if not train_embeds:
            self.embedding.weight.requires_grad = False
        self.rnn = nn.GRU(emb_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout)
        self.full_layer = nn.Linear(hidden_size, out_size)
        
        # self.rnn = [nn.GRU(emb_size if l == 0 else hidden_size, hidden_size if l != num_layers-1 else out_size, 1, batch_first=True, dropout=dropout) for l in range(num_layers)]
    
    def forward(self, input, input_lengths):
        emb = self.embedding(input)
        packed_emb = nn.utils.rnn.pack_padded_sequence(emb, input_lengths, batch_first=True, enforce_sorted=False)
        rnn_output, hidden = self.rnn(packed_emb)
        # print(nn.utils.rnn.pad_packed_sequence(rnn_output).size())
        # print(hidden.size())
        # print(hidden[-1].size())

        output = self.full_layer(hidden[-1])
        return output
        