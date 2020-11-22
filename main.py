import os
import math
import time
import torch
import torch.nn as nn
from data.data import get_data_loader, get_node_embeddings
from model import RNNModel

args = {
    "embedding_size": 128,
    "hidden_size": 512,
    "num_layers": 3,
    "dropout": 0.3,
    "batch_size": 32,
    "eval_batch_size": 32,
    "test_batch_size": 32,
    "lr": 0.0001,
    "wdecay": 1.2e-6,
    "epochs": 50,
    "seed": 41,
    "train_embeds": False,
    "cuda": True,
    "log_interval": 300,
}

model_save_path = "model.pt"
device = 'cuda:0' if args["cuda"] else 'cpu'

def train(epoch, model, data_loader, criterion, optimizer, log_interval=100):
    total_loss = 0
    start_time = time.time()
    batch = 0
    model.train()
    for (input, input_lengths), targets in data_loader:
        batch += 1
        if args["cuda"]:
            input, targets = input.cuda(), targets.cuda()
        optimizer.zero_grad()
        output = model(input,input_lengths)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.data
        if batch%log_interval == 0:
            cur_loss = total_loss.item() / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f}'.format(
                epoch, batch, len(data_loader), optimizer.param_groups[0]['lr'],
                elapsed * 1000 / log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()

def evaluate(model, criterion, data_loader):
    model.eval()
    total_loss = 0
    for (input, input_lengths), targets in data_loader:
        if args["cuda"]:
            input, targets = input.cuda(), targets.cuda()
        output = model(input, input_lengths)
        total_loss += criterion(output, targets).data
    return total_loss/len(data_loader)

if __name__ == '__main__':
    data_path = "data"
    train_dataloader, valid_dataloader, test_dataloader = get_data_loader(data_path, args)
    node_embeddings, padding_idx = get_node_embeddings(os.path.join(data_path, "node_embeddings.npy"))
    model = RNNModel(node_embeddings, padding_idx, args["embedding_size"], args["hidden_size"], 
                    len(node_embeddings)-1, args["num_layers"], args["dropout"], args["train_embeds"])
    print(model)

    criterion = nn.CrossEntropyLoss()
    if args["cuda"]:
        model = model.cuda()
        criterion = criterion.cuda()

    if args["train_embeds"]:
        parameters = model.parameters()
    else:
        parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, args["lr"], weight_decay=args["wdecay"])

    stored_loss = 100000000
    for e in range(1, args["epochs"]+1):
        print("="*89)
        print("epoch : ", e)
        epoch_start_time = time.time()
        train(e, model, train_dataloader, criterion, optimizer, args["log_interval"])
        val_loss = evaluate(model, criterion, valid_dataloader)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f}'.format(
            e, (time.time() - epoch_start_time), val_loss))
        print('-' * 89)
        if val_loss < stored_loss:
            with open(model_save_path, 'wb') as f:
                torch.save(model.state_dict(), f)
            print('Saving model (new best validation)')
            stored_loss = val_loss
        print("="*89)

    if os.path.exists(model_save_path):
        model_state_dict = torch.load(model_save_path, map_location=device)
        model.load_state_dict(model_state_dict)
        test_loss = evaluate(model, criterion, test_dataloader)
        print('=' * 89)
        print('| End of training | test loss {:5.2f}'.format(test_loss))
        print('=' * 89)