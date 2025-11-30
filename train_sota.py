# train_sota.py  (version: compat-fixed)
import argparse, os, math, random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler

from torchvision import datasets, transforms, models
from pretrained_loader import init_for_training  # your helper

# ---------- utils ----------
def seed_everything(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def _safe_weights_and_norm(arch: str):
    """
    Return (weights_obj_or_None, mean, std) in a version-compatible way.
    If weights/meta not available, fall back to ImageNet mean/std.
    """
    arch = arch.lower()
    weights = None
    try:
        if arch == "convnext_base":
            weights = models.ConvNeXt_Base_Weights.DEFAULT
        elif arch == "efficientnet_v2_m":
            weights = models.EfficientNet_V2_M_Weights.DEFAULT
        elif arch == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
        elif arch == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT
        else:
            weights = models.ResNet18_Weights.DEFAULT
    except Exception:
        weights = None

    mean, std = IMAGENET_MEAN, IMAGENET_STD
    # torchvision ≥0.13 often has weights.meta, but not always
    try:
        if weights is not None and hasattr(weights, "meta"):
            meta = getattr(weights, "meta", {})
            mean = meta.get("mean", IMAGENET_MEAN)
            std  = meta.get("std", IMAGENET_STD)
    except Exception:
        pass

    return weights, mean, std


def build_transforms(arch: str, img_size: int):
    weights, mean, std = _safe_weights_and_norm(arch)

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tfms, val_tfms


def make_weighted_sampler(dataset: datasets.ImageFolder):
    labels = [y for _, y in dataset.samples]
    class_counts = Counter(labels)
    total = sum(class_counts.values())
    class_weight = {c: total / (len(class_counts) * cnt) for c, cnt in class_counts.items()}
    sample_weights = [class_weight[y] for _, y in dataset.samples]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def mixup_data(x, y, num_classes, alpha=0.2):
    if alpha <= 0.0:
        return x, torch.nn.functional.one_hot(y, num_classes=num_classes).float(), 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_onehot = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
    mixed_y = lam * y_onehot + (1 - lam) * y_onehot[index, :]
    return mixed_x, mixed_y, lam


def soft_ce_loss(logits, soft_targets):
    log_prob = torch.log_softmax(logits, dim=1)
    return -(soft_targets * log_prob).sum(dim=1).mean()


def train_one_epoch(model, loader, optimizer, scaler, device, num_classes, mixup_alpha, freeze_bn):
    model.train()
    if freeze_bn:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()

    running = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        xb, soft_yb, _ = mixup_data(xb, yb, num_classes, alpha=mixup_alpha)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(xb)
            loss = soft_ce_loss(logits, soft_yb)

        scaler.scale(loss).backward()   # <- fixed
        scaler.step(optimizer)
        scaler.update()
        running += loss.item() * xb.size(0)

    return running / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct / max(total, 1)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=384)  # 336–384 recommended
    ap.add_argument("--arch", default="convnext_base",
                    choices=["resnet18","resnet34","resnet50","convnext_base","efficientnet_v2_m"])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--freeze-epochs", type=int, default=1)
    ap.add_argument("--out", default="runs/sota")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    train_tfms, val_tfms = build_transforms(args.arch, args.img_size)
    d_train = datasets.ImageFolder(Path(args.data) / "train", transform=train_tfms)
    d_val   = datasets.ImageFolder(Path(args.data) / "val",   transform=val_tfms)

    # Model
    model, _, arch = init_for_training(args.arch, classes=d_train.classes)
    model = model.to(device)

    # Separate head vs. backbone for freezing and LR
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if any(k in name for k in ["classifier.2", "classifier.1", "fc"]):
            head_params.append(p)
        else:
            backbone_params.append(p)
    for p in backbone_params:
        p.requires_grad_(False)

    optimizer = optim.AdamW([
        {"params": head_params, "lr": args.lr},
        {"params": backbone_params, "lr": args.lr * 0.1}
    ], weight_decay=1e-4)
    scaler = GradScaler()
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs))

    # Loaders
    sampler = make_weighted_sampler(d_train)
    train_loader = DataLoader(d_train, batch_size=args.batch_size, sampler=sampler, num_workers=2)
    val_loader   = DataLoader(d_val,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    os.makedirs(args.out, exist_ok=True)
    best = 0.0
    patience, bad = 7, 0

    for epoch in range(1, args.epochs + 1):
        # Unfreeze after short freeze
        if epoch == args.freeze_epochs + 1:
            for p in backbone_params:
                p.requires_grad_(True)

        # Warmup LR for first few epochs
        if epoch <= args.warmup_epochs:
            for i, g in enumerate(optimizer.param_groups):
                base = args.lr if i == 0 else args.lr * 0.1
                g["lr"] = base * epoch / max(1, args.warmup_epochs)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            num_classes=len(d_train.classes), mixup_alpha=args.mixup,
            freeze_bn=(epoch <= args.freeze_epochs)
        )

        val_acc = evaluate(model, val_loader, device)
        if epoch > args.warmup_epochs:
            scheduler.step()

        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best:
            best = val_acc; bad = 0
            torch.save({
                "model_state": model.state_dict(),
                "classes": d_train.classes,
                "arch": arch
            }, os.path.join(args.out, "best.pth"))
            print("Saved checkpoint.")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

if __name__ == "__main__":
    main()
