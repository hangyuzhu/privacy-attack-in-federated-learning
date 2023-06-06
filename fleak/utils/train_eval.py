import torch


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    # batch training
    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)
        # clear tensor grad
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        # this step also update the optimizer.state, i.e momentum state
        # the momentum state can be extracted by optimizer.state_dict()
        optimizer.step()


def evaluate(model, device, eval_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for features, labels in eval_loader:
            features, labels = features.to(device), labels.to(device)
            output = model(features)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
    return correct
