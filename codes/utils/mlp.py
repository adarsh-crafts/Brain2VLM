import argparse, os, gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from himalaya.scoring import correlation_score
from sklearn.preprocessing import StandardScaler
import psutil


# Memory monitor
def mem():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, dim=2048):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        identity = x
        out = self.fc(x)
        out = self.gelu(out)
        out = self.norm(out)
        return out + identity


# MLP decoder
class MLPDecoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.GELU(),
            nn.LayerNorm(2048),
        )

        self.res1 = ResidualBlock(2048)
        self.res2 = ResidualBlock(2048)

        self.output = nn.Linear(2048, out_dim)

    def forward(self, x):
        x = self.input(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.output(x)
        return x


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--target", type=str, default='')
    parser.add_argument("--roi", required=True, type=str, nargs="*")
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)

    opt = parser.parse_args()
    target = opt.target
    roi = opt.roi
    subject = opt.subject

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    mridir = f'../../mrifeat/{subject}/'
    featdir = '../../nsdfeat/subjfeat/'
    savedir = f'../../decoded/{subject}/'
    os.makedirs(savedir, exist_ok=True)

    # Load X
    X_list = []
    X_te_list = []

    for croi in roi:
        if 'conv' in target:
            arr = np.load(f'{mridir}/{subject}_{croi}_betas_ave_tr.npy', mmap_mode='r')
        else:
            arr = np.load(f'{mridir}/{subject}_{croi}_betas_tr.npy', mmap_mode='r')
        X_list.append(arr.astype("float32"))

        arr_te = np.load(f'{mridir}/{subject}_{croi}_betas_ave_te.npy', mmap_mode='r')
        X_te_list.append(arr_te.astype("float32"))

    # Stack training data
    total_dim = sum(arr.shape[1] for arr in X_list)
    n_samples = X_list[0].shape[0]

    X = np.empty((n_samples, total_dim), dtype=np.float32)
    start = 0
    for arr in X_list:
        end = start + arr.shape[1]
        X[:, start:end] = arr
        start = end
    del X_list

    # Stack test data
    total_dim_te = sum(arr.shape[1] for arr in X_te_list)
    n_samples_te = X_te_list[0].shape[0]

    X_te = np.empty((n_samples_te, total_dim_te), dtype=np.float32)
    start = 0
    for arr in X_te_list:
        end = start + arr.shape[1]
        X_te[:, start:end] = arr
        start = end
    del X_te_list

    print(f'[MEM] after X/X_te load: {mem():.2f} GB')

    # Load Y
    Y_raw = np.load(f'{featdir}/{subject}_each_{target}_tr.npy', mmap_mode='r')
    Y = Y_raw.reshape([X.shape[0], -1]).astype("float32")

    Y_te_raw = np.load(f'{featdir}/{subject}_ave_{target}_te.npy', mmap_mode='r')
    Y_te = Y_te_raw.reshape([X_te.shape[0], -1]).astype("float32")

    print(f'[MEM] after Y load: {mem():.2f} GB')

    # Standardize X
    scaler = StandardScaler(with_mean=True, with_std=True)
    X = scaler.fit_transform(X)
    X_te = scaler.transform(X_te)

    # Convert to torch
    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)

    dataset = TensorDataset(X_t, Y_t)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    # Build model
    model = MLPDecoder(X.shape[1], Y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    print(f'Now training MLP for {subject}: {roi}, {target}')
    print(f'X {X.shape}, Y {Y.shape}, X_te {X_te.shape}')
    print(f'[MEM] before training: {mem():.2f} GB')

    # Training loop
    for epoch in range(opt.epochs):
        model.train()
        total_loss = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{opt.epochs} | Loss: {total_loss/len(loader):.6f}")

    print(f'[MEM] after training: {mem():.2f} GB')

    # Prediction
    model.eval()
    with torch.no_grad():
        X_te_t = torch.from_numpy(X_te).to(device)
        scores = model(X_te_t).cpu().numpy()

    print(f'[MEM] after predict: {mem():.2f} GB')

    # Correlation score
    rs = correlation_score(Y_te.T, scores.T)
    print(f'Prediction accuracy is: {np.mean(rs):3.3f}')

    np.save(f'{savedir}/{subject}_{"_".join(roi)}_scores_{target}_mlp.npy', scores)


if __name__ == "__main__":
    main()