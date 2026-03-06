import argparse, os, gc, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from himalaya.scoring import correlation_score
from sklearn.preprocessing import StandardScaler
import psutil


def mem():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        out = self.fc(x)
        out = self.gelu(out)
        out = self.norm(out)
        out = self.dropout(out)
        return out + identity


class MLPDecoder(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=2048, depth=2, dropout=0.2):
        super().__init__()

        self.input = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(depth)]
        )

        self.output = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):

        x = self.input(x)

        for block in self.blocks:
            x = block(x)

        x = self.output(x)

        return x


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--target", type=str, default='')
    parser.add_argument("--roi", required=True, type=str, nargs="*")
    parser.add_argument("--subject", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)

    # ablation params
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--data_frac", type=float, default=1.0)

    opt = parser.parse_args()

    set_seed(opt.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mridir = f'../../mrifeat/{opt.subject}/'
    featdir = '../../nsdfeat/subjfeat/'

    base_dir = f'../../decoded/{opt.subject}/'
    model_dir = os.path.join(base_dir, "models")

    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(
        model_dir,
        f"best_mlp_d{opt.depth}_w{opt.hidden_dim}_frac{opt.data_frac}.pt"
    )

    # ------------------------
    # Load X
    # ------------------------

    X_list, X_te_list = [], []

    for croi in opt.roi:

        if 'conv' in opt.target:
            arr = np.load(
                f'{mridir}/{opt.subject}_{croi}_betas_ave_tr.npy',
                mmap_mode='r'
            )
        else:
            arr = np.load(
                f'{mridir}/{opt.subject}_{croi}_betas_tr.npy',
                mmap_mode='r'
            )

        X_list.append(arr.astype("float32"))

        arr_te = np.load(
            f'{mridir}/{opt.subject}_{croi}_betas_ave_te.npy',
            mmap_mode='r'
        )

        X_te_list.append(arr_te.astype("float32"))

    total_dim = sum(arr.shape[1] for arr in X_list)

    n_samples = X_list[0].shape[0]

    X = np.empty((n_samples, total_dim), dtype=np.float32)

    start = 0

    for arr in X_list:
        end = start + arr.shape[1]
        X[:, start:end] = arr
        start = end

    del X_list

    total_dim_te = sum(arr.shape[1] for arr in X_te_list)

    n_samples_te = X_te_list[0].shape[0]

    X_te = np.empty((n_samples_te, total_dim_te), dtype=np.float32)

    start = 0

    for arr in X_te_list:
        end = start + arr.shape[1]
        X_te[:, start:end] = arr
        start = end

    del X_te_list

    print(f'[MEM] after X load: {mem():.2f} GB')

    # ------------------------
    # Load targets
    # ------------------------

    Y = np.load(
        f'{featdir}/{opt.subject}_each_{opt.target}_tr.npy',
        mmap_mode='r'
    )

    Y = Y.reshape([X.shape[0], -1]).astype("float32")

    Y_te = np.load(
        f'{featdir}/{opt.subject}_ave_{opt.target}_te.npy',
        mmap_mode='r'
    )

    Y_te = Y_te.reshape([X_te.shape[0], -1]).astype("float32")

    # ------------------------
    # Normalize
    # ------------------------

    scaler = StandardScaler()

    X = scaler.fit_transform(X)
    X_te = scaler.transform(X_te)

    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)

    dataset = TensorDataset(X_t, Y_t)

    # ------------------------
    # Data fraction ablation
    # ------------------------

    if opt.data_frac < 1.0:

        subset_size = int(len(dataset) * opt.data_frac)

        dataset, _ = random_split(
            dataset,
            [subset_size, len(dataset) - subset_size],
            generator=torch.Generator().manual_seed(opt.seed)
        )

        print(f"Using {subset_size} samples ({opt.data_frac*100:.0f}%)")

    # ------------------------
    # Train/Val split
    # ------------------------

    val_size = int(len(dataset) * opt.val_split)

    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(opt.seed)
    )

    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=opt.batch_size, shuffle=False)

    # ------------------------
    # Model
    # ------------------------

    model = MLPDecoder(
        X.shape[1],
        Y.shape[1],
        hidden_dim=opt.hidden_dim,
        depth=opt.depth
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        weight_decay=opt.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=opt.epochs
    )

    criterion = nn.MSELoss()

    best_val = float("inf")
    patience_counter = 0

    print(f"Training: {opt.subject} | ROI {opt.roi} | target {opt.target}")
    print(f"Train: {train_size} | Val: {val_size}")

    # ------------------------
    # Training
    # ------------------------

    for epoch in range(opt.epochs):

        model.train()
        train_loss = 0

        for xb, yb in train_loader:

            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)

            loss = criterion(pred, yb)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()

        val_loss = 0

        with torch.no_grad():

            for xb, yb in val_loader:

                xb, yb = xb.to(device), yb.to(device)

                pred = model(xb)

                val_loss += criterion(pred, yb).item()

        val_loss /= len(val_loader)

        scheduler.step()

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train {train_loss:.6f} | "
            f"Val {val_loss:.6f} | "
            f"LR {scheduler.get_last_lr()[0]:.6e}"
        )

        if val_loss < best_val:

            best_val = val_loss
            patience_counter = 0

            torch.save(model.state_dict(), model_path)

        else:

            patience_counter += 1

            if patience_counter >= opt.patience:
                print("Early stopping triggered.")
                break

    # ------------------------
    # Evaluation
    # ------------------------

    model.load_state_dict(torch.load(model_path))

    model.eval()

    with torch.no_grad():

        X_te_t = torch.from_numpy(X_te).to(device)

        scores = model(X_te_t).cpu().numpy()

    rs = correlation_score(Y_te.T, scores.T)

    print(f"Final correlation: {np.mean(rs):.4f}")

    np.save(
        f'../../decoded/{opt.subject}/{opt.subject}_{"_".join(opt.roi)}_scores_{opt.target}_mlp_d{opt.depth}_w{opt.hidden_dim}_frac{opt.data_frac}.npy',
        scores
    )


if __name__ == "__main__":
    main()