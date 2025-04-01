import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def top_k_accuracy(output, target, k=3):
    top_k = torch.topk(output, k, dim=1).indices
    correct = 0
    for i in range(output.size(0)):
        if target[i].nonzero().view(-1).tolist():
            if any(g in top_k[i] for g in target[i].nonzero().view(-1).tolist()):
                correct += 1
    return correct / output.size(0)

def multilabel_accuracy(output, target, threshold=0.5):
    pred = (torch.sigmoid(output) > threshold).float()
    correct = (pred == target).float().mean()
    return correct.item()

def evaluate(model, data_loader, device):
    model.eval()
    total_loss, acc, top_k_acc = 0, 0, 0
    criterion = torch.nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data.x, data.edge_index)
            loss = criterion(output, data.y)
            total_loss += loss.item()
            acc += multilabel_accuracy(output, data.y)
            top_k_acc += top_k_accuracy(output, data.y, k=3)
    
    n = len(data_loader)
    return total_loss / n, acc / n, top_k_acc / n
