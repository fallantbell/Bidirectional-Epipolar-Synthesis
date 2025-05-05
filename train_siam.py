import os 
import torch
import torch.optim as optim
import math
from tqdm import tqdm
from einops import rearrange

def train(model, train_loader, folder_logs, folder_model, num_epochs=20, lr=1e-4, betas=(0.9,0.95), wd=0.05, warmup_epoch=20):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(num_epochs)):
        for batch_idx, (data) in enumerate(train_loader):
            data = data['rgbs'].to(device)
            optimizer.zero_grad()
            loss, pred = model(data)
            loss = loss.mean() 
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")
        path = os.path.join(folder_model, f'epoch_{epoch}.pt')
        torch.save(model.state_dict(), path)
        with open(folder_logs, 'a+') as f:
            f.writelines(f"Epoch: {epoch}/{num_epochs}, Loss: {loss.item()} \n")
    print("Training complete!")
    return model