# train_resnet.py  (UPGRADED)
import argparse, os
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from pretrained_loader import init_for_training  # NEW

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--arch", default="convnext_base",
                    choices=["resnet18","resnet34","resnet50","convnext_base","efficientnet_v2_m"])
    ap.add_argument("--out", default="runs/resnet")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tfm_train = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor()
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])

    train_ds = datasets.ImageFolder(Path(args.data)/"train", transform=tfm_train)
    val_ds   = datasets.ImageFolder(Path(args.data)/"val",   transform=tfm_eval)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ---- PRETRAINED BACKBONE HERE ----
    model, _, arch = init_for_training(args.arch, classes=train_ds.classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

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

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total   += yb.size(0)
        acc = (correct / total) if total else 0.0
        print(f"Val Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": train_ds.classes,
                "arch": arch
            }, os.path.join(args.out, "best.pth"))
            print("Saved checkpoint.")

if __name__ == "__main__":
    main()
