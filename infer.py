import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# ----------------------------
# Config
# ----------------------------
IMG_SIZE_DEFAULT = 224
VALID_IMG_EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

# ----------------------------
# Model loading (ResNet only)
# ----------------------------
ARCH_CHOICES = {"resnet18", "resnet34", "resnet50"}


def build_resnet_head(arch: str, num_classes: int) -> nn.Module:
    if arch == "resnet18":
        model = models.resnet18(weights=None)
    elif arch == "resnet34":
        model = models.resnet34(weights=None)
    elif arch == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported arch '{arch}'. Choose from {sorted(ARCH_CHOICES)}")
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model


def load_resnet_checkpoint(weights_path: Path, force_arch: Optional[str] = None) -> Tuple[nn.Module, List[str], str]:
    ckpt = torch.load(weights_path, map_location="cpu")
    classes: List[str] = ckpt.get("classes")
    if not classes:
        raise RuntimeError("Checkpoint missing 'classes' list.")

    arch = ckpt.get("arch", "resnet18")
    if force_arch is not None:
        arch = force_arch
    if arch not in ARCH_CHOICES:
        raise RuntimeError(
            f"Checkpoint arch='{arch}' not in {sorted(ARCH_CHOICES)}. Use --arch to force if needed.")

    model = build_resnet_head(arch, num_classes=len(classes))
    state = ckpt.get("model_state")
    if state is None:
        raise RuntimeError("Checkpoint missing 'model_state'.")
    missing = model.load_state_dict(state, strict=True)
    del missing  # not used; strict=True will already error if mismatch
    model.eval()
    return model, classes, arch

# ----------------------------
# Grad‑CAM (for ResNet layer4 last block)
# ----------------------------
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._h1 = target_layer.register_forward_hook(self._forward_hook)
        self._h2 = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1))
        score = logits[:, class_idx]
        score.backward()
        # weights = global-average-pooled gradients
        w = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (w * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_cam_on_image(img_pil: Image.Image, cam, alpha: float = 0.45) -> Image.Image:
    import matplotlib.pyplot as plt
    import io

    cam_uint8 = (cam * 255).astype("uint8")
    fig = plt.figure(frameon=False)
    fig.set_size_inches(2, 2)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.set_axis_off()
    ax.imshow(cam_uint8, cmap="jet", interpolation="nearest")
    buf = io.BytesIO()
    fig.canvas.print_png(buf)
    plt.close(fig)

    heatmap = Image.open(buf).convert("RGBA").resize(img_pil.size, Image.BILINEAR)
    base = img_pil.convert("RGBA")
    out = Image.blend(base, heatmap, alpha=alpha)
    return out.convert("RGB")

# ----------------------------
# Inference utils
# ----------------------------

def make_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def predict_folder(model: nn.Module, classes: List[str], input_dir: Path, img_size: int,
                   save_cam_dir: Optional[Path] = None) -> list:
    tfm = make_transform(img_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Prepare Grad‑CAM if requested
    cam_engine = None
    if save_cam_dir is not None:
        if not hasattr(model, "layer4"):
            raise RuntimeError("Grad‑CAM requested but model is not a ResNet (no layer4).")
        cam_engine = GradCAM(model, model.layer4[-1])
        save_cam_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for p in sorted(input_dir.rglob("*")):
        if p.suffix.lower() not in VALID_IMG_EXTS:
            continue
        img = Image.open(p).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            conf, idx = torch.max(probs, dim=0)
        pred_name = classes[int(idx)]
        rec = {
            "image": str(p),
            "pred_class": pred_name,
            "confidence": float(conf.item()),
        }
        if cam_engine is not None:
            cam = cam_engine(x, int(idx))
            overlay = overlay_cam_on_image(img.resize((img_size, img_size)), cam)
            cam_path = save_cam_dir / f"{p.stem}_cam.jpg"
            overlay.save(cam_path)
            rec["cam_image"] = str(cam_path)
        results.append(rec)
    return results

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="ResNet inference with optional Grad‑CAM")
    # Short flags (quick use)
    ap.add_argument("-w", "--weights", required=False, help="Path to ResNet checkpoint .pth")
    ap.add_argument("-i", "--input", required=False, help="Folder of images")
    ap.add_argument("-o", "--out", default="outputs/preds.json", help="Output JSON path")
    ap.add_argument("-g", "--gradcam", action="store_true", help="Save Grad‑CAM overlays next to JSON")

    # Full flags / advanced
    ap.add_argument("--arch", choices=sorted(ARCH_CHOICES), default=None, help="Force ResNet arch (overrides ckpt)")
    ap.add_argument("--img-size", type=int, default=IMG_SIZE_DEFAULT)

    args = ap.parse_args()

    # Defaults to common training output if not provided
    weights_path = Path(args.weights or "runs/resnet/best.pth")
    input_dir = Path(args.input or "sample_inputs")

    if not weights_path.exists():
        raise SystemExit(f"Weights not found: {weights_path}")
    if not input_dir.exists():
        raise SystemExit(f"Input folder not found: {input_dir}")

    model, classes, arch = load_resnet_checkpoint(weights_path, force_arch=args.arch)

    cam_dir = None
    if args.gradcam:
        cam_dir = Path(args.out).with_suffix("").parent / "cam"

    input_dir = input_dir.resolve()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    results = predict_folder(model, classes, input_dir, args.img_size, save_cam_dir=cam_dir)
    Path(args.out).write_text(json.dumps(results, indent=2))

    print(f"✅ Saved predictions to {args.out}")
    if cam_dir is not None:
        print(f"✅ Saved Grad‑CAM overlays to {cam_dir}")


if __name__ == "__main__":
    main()
