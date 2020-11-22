import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

padding_idx = 27770
# print(padding_idx)

class GraphDataset(Dataset):
    def __init__(self, file_path):
        self.data = self.parse(file_path)

    def parse(self, file_path):
        data = []
        with open(file_path, "r") as file:
            file_lines = file.readlines()
            for file_line in file_lines:
                file_line_split = file_line.strip().split()
                file_line_split = list(map(int, file_line_split))
                data.append(file_line_split)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inp = torch.LongTensor(self.data[index][1:])
        inp_len = torch.LongTensor([len(inp)])
        out = torch.LongTensor([self.data[index][0]])

        return inp, inp_len, out

def collate_fn(data):
    inps, inp_lens, outs = zip(*data)
    input_lengths = torch.stack(inp_lens, 0).squeeze(1)
    lengths = list(map(len, inps))
    inputs = torch.ones(len(inps), max(lengths)).long()*padding_idx
    for i, inp in enumerate(inps):
        end = lengths[i]
        inputs[i, :end] = inp[:end]

    targets = torch.LongTensor(outs)

    return (inputs, input_lengths), targets

def get_data_loader(path, args):
    train_dataset = GraphDataset(os.path.join(path, "train.txt"))
    valid_dataset = GraphDataset(os.path.join(path, "valid.txt"))
    test_dataset = GraphDataset(os.path.join(path, "test.txt"))

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args['batch_size'],
                                 shuffle=True, num_workers=4, drop_last=True,
                                 collate_fn=collate_fn, pin_memory=True)

    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args['eval_batch_size'],
                                 shuffle=True, num_workers=4, drop_last=True,
                                 collate_fn=collate_fn, pin_memory=True)

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args['test_batch_size'],
                                 shuffle=True, num_workers=4, drop_last=True,
                                 collate_fn=collate_fn, pin_memory=True)

    return train_dataloader, valid_dataloader, test_dataloader

def get_node_embeddings(path):
    return torch.FloatTensor(np.load(path)), padding_idx


if __name__ == '__main__':
    args = {
        "batch_size": 8,
        "eval_batch_size": 8,
        "test_batch_size": 8
    }
    train_dataloader, _, _ = get_data_loader(".", args)
    x, y = next(iter(train_dataloader))
    print(y)
    print(y.size())
    print(x[0].size())
    print(x[0])
    print(x[1].size())