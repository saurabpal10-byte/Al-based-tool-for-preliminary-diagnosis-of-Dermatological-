# gradcam.py
# Works with ConvNeXt-Base, EfficientNet-V2-M and ResNet backbones using your new pretrained loader.
# Saves: overlay PNG, raw CAM (grayscale PNG), and a JSON with prediction+confidence for each image.

import argparse, os, json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Uses your new helper that knows how to load any supported backbone and expose the right target layer
from pretrained_loader import load_from_checkpoint  # <- provided in the upgrade


# ----------------------------
# Image utilities
# ----------------------------
def load_image(path):
    img = Image.open(path).convert("RGB")
    return img

def preprocess(img: Image.Image, size=224):
    # Simple, deterministic resize + toTensor + batch
    img = img.resize((size, size), Image.BICUBIC)
    x = np.asarray(img).astype(np.float32) / 255.0  # HWC, [0,1]
    x = np.transpose(x, (2, 0, 1))                  # CHW
    x = torch.from_numpy(x).unsqueeze(0)            # 1CHW
    return x

def to_pil(img_np):
    # img_np float in [0,1], HWC
    img_np = np.clip(img_np, 0.0, 1.0)
    img_u8 = (img_np * 255).astype(np.uint8)
    return Image.fromarray(img_u8)


# ----------------------------
# Grad-CAM core
# ----------------------------
class GradCAM:
    """
    Vanilla Grad-CAM:
      - Forward hook grabs feature maps from target layer
      - Backward hook grabs gradients wrt that layer
      - Weigh channels by global-average of gradients
      - ReLU, normalize to [0,1], upsample to input size
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module, device="cpu"):
        self.model = model
        self.target_layer = target_layer
        self.device = device

        self.fmap = None
        self.grad = None
        self._fwd_hook = self.target_layer.register_forward_hook(self._save_fwd)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_bwd)

    def _save_fwd(self, module, inp, out):
        # out: [N, C, H, W]
        self.fmap = out.detach()

    def _save_bwd(self, module, grad_input, grad_output):
        # grad_output: tuple with [grad wrt out]
        self.grad = grad_output[0].detach()

    def __call__(self, logits: torch.Tensor, class_idx: int, input_size_hw):
        """
        logits: model(x) BEFORE softmax, shape [1, num_classes]
        class_idx: int
        input_size_hw: (H, W) of the model input to upsample CAM to
        """
        # Clear previous grads
        self.model.zero_grad(set_to_none=True)

        # Compute scalar for class of interest and backprop
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        # Feature maps and gradients captured by hooks
        fmap = self.fmap                          # [1, C, h, w]
        grad = self.grad                          # [1, C, h, w]
        if fmap is None or grad is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        # Global-average the gradients spatially: weights [C]
        weights = torch.mean(grad, dim=(2, 3))[0]  # [C]

        # Weighted sum of channels
        cam = torch.zeros(fmap.shape[2:], dtype=torch.float32, device=fmap.device)  # [h, w]
        for c, w in enumerate(weights):
            cam += w * fmap[0, c, :, :]

        # ReLU
        cam = F.relu(cam)

        # Normalize to [0,1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        # Upsample to input size
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=input_size_hw, mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()  # H, W in [0,1]
        return cam

    def remove_hooks(self):
        try:
            self._fwd_hook.remove()
        except Exception:
            pass
        try:
            self._bwd_hook.remove()
        except Exception:
            pass


# ----------------------------
# Visualization
# ----------------------------
def colorize_cam(gray_cam):
    """
    Convert a [H,W] cam in [0,1] to RGB heatmap (matplotlib-free).
    We'll use a simple jet-like ramp for portability without extra deps.
    """
    h, w = gray_cam.shape
    cam = np.zeros((h, w, 3), dtype=np.float32)

    # Jet-ish colormap constructed from piecewise ramps
    def jet(v):
        # v in [0,1]
        r = np.clip(1.5 * v - 0.5, 0, 1)
        g = np.clip(1.5 - 1.5 * np.abs(v - 2/3), 0, 1)  # simple bell-ish
        b = np.clip(1.5 - 1.5 * v, 0, 1)
        return r, g, b

    r, g, b = jet(gray_cam)
    cam[..., 0] = r
    cam[..., 1] = g
    cam[..., 2] = b
    return cam  # float RGB [0,1]

def overlay_cam_on_image(img_pil: Image.Image, cam_gray: np.ndarray, alpha=0.35):
    """
    Blend original image with heatmap. alpha is the CAM weight.
    """
    img_np = np.asarray(img_pil).astype(np.float32) / 255.0  # HWC
    heat = colorize_cam(cam_gray)                            # HWC [0,1]
    overlay = (1 - alpha) * img_np + alpha * heat
    overlay = np.clip(overlay, 0.0, 1.0)
    return to_pil(overlay)

def save_gray_cam(cam_gray: np.ndarray, path: Path):
    cam_img = (np.clip(cam_gray, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(cam_img).save(path)


# ----------------------------
# Prediction + Grad-CAM for one image
# ----------------------------
def run_one(model, target_layer, classes, img_path, device, img_size=224, alpha=0.35):
    model.eval()
    img = load_image(img_path)
    x = preprocess(img, size=img_size).to(device)

    with torch.no_grad():
        logits = model(x)  # [1, K]
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
        idx = idx.item()
        conf = float(conf.item())

    # Grad-CAM requires gradients => forward again without no_grad
    logits = model(x)
    cam_engine = GradCAM(model, target_layer, device=device)
    cam_gray = cam_engine(logits, idx, input_size_hw=(img_size, img_size))
    cam_engine.remove_hooks()

    # Build visuals
    overlay = overlay_cam_on_image(img.resize((img_size, img_size), Image.BICUBIC), cam_gray, alpha=alpha)
    return {
        "pred_class": classes[idx],
        "confidence": conf,
        "cam_gray": cam_gray,
        "overlay_pil": overlay
    }


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to classifier checkpoint .pth")
    ap.add_argument("--input", required=True, help="Image file OR folder containing images")
    ap.add_argument("--out", default="outputs/gradcam", help="Output folder")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--alpha", type=float, default=0.35, help="Blend weight for CAM overlay")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model + correct target layer + classes based on the checkpoint (works for convnext/efficientnet/resnet)
    model, target_layer, classes, arch = load_from_checkpoint(args.weights)  # <- key integration
    model = model.to(device)

    in_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    if in_path.is_file():
        img_paths = [in_path]
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        img_paths = [p for p in in_path.rglob("*") if p.suffix.lower() in exts]

    results = []
    for p in img_paths:
        try:
            out = run_one(model, target_layer, classes, p, device, img_size=args.img_size, alpha=args.alpha)
            # Save files
            stem = p.stem
            overlay_path = out_dir / f"{stem}_overlay.png"
            cam_path = out_dir / f"{stem}_cam.png"

            out["overlay_pil"].save(overlay_path)
            save_gray_cam(out["cam_gray"], cam_path)

            results.append({
                "image": str(p),
                "pred_class": out["pred_class"],
                "confidence": out["confidence"],
                "overlay": str(overlay_path),
                "cam": str(cam_path)
            })
            print(f"[OK] {p.name}: {out['pred_class']} ({out['confidence']:.3f})")
        except Exception as e:
            print(f"[ERR] {p}: {e}")

    # Write a summary JSON
    (out_dir / "summary.json").write_text(json.dumps({
        "arch": arch,
        "num_images": len(img_paths),
        "results": results
    }, indent=2))
    print(f"Saved Grad-CAM outputs to: {out_dir}")

if __name__ == "__main__":
    main()
