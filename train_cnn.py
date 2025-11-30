import argparse, os, time
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, num_classes))
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data_splits directory")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="runs/cnn")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tfm_train = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])
    train_ds = datasets.ImageFolder(Path(args.data)/"train", transform=tfm_train)
    val_ds = datasets.ImageFolder(Path(args.data)/"val", transform=tfm_eval)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = SimpleCNN(num_classes=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs+1):
        model.train()
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # val
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds==yb).sum().item()
                total += yb.size(0)
        acc = correct/total if total>0 else 0.0
        print(f"Val Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": train_ds.classes
            }, os.path.join(args.out, "best.pth"))
            print("Saved checkpoint.")

if __name__ == "__main__":
    main()