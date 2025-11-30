# app_web.py — Gradio (auto-loads weights + saves Grad-CAM)
# One click: python app_web.py

from pathlib import Path
from typing import Tuple
import time
import gradio as gr
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np

# ---------------- Config ----------------
IMG_SIZE = 224
DEFAULT_WEIGHTS = Path("runs/resnet/best.pth")
SAVE_DIR = Path("web_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ------------- Model helpers -------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def detect_ckpt_type(ckpt):
    if "arch" in ckpt:
        a = str(ckpt["arch"]).lower()
        if a in {"resnet18", "resnet34", "resnet50"}:
            return "resnet", a
    keys = list(ckpt.get("model_state", {}).keys())
    if any(k.startswith("layer1.") or k.startswith("fc.") for k in keys):
        return "resnet", "resnet18"
    if any(k.startswith("features.0") or k.startswith("classifier.1") for k in keys):
        return "cnn", None
    return "resnet", "resnet18"


def load_model(weights_path: Path):
    ckpt = torch.load(weights_path, map_location="cpu")
    classes = ckpt.get("classes")
    if not classes:
        raise RuntimeError("Checkpoint missing 'classes'.")
    mtype, arch = detect_ckpt_type(ckpt)

    if mtype == "resnet":
        if arch == "resnet34":
            model = models.resnet34(weights=None)
        elif arch == "resnet50":
            model = models.resnet50(weights=None)
        else:
            model = models.resnet18(weights=None)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, len(classes))
    else:
        model = SimpleCNN(num_classes=len(classes))

    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model, classes, mtype, (arch or "cnn")


# ------------- Grad-CAM -------------
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer
        self.acts = None
        self.grads = None
        self.target_layer.register_forward_hook(self._fwd)
        self.target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.acts = o.detach()

    def _bwd(self, m, gin, gout):
        self.grads = gout[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, 1).item())
        logits[:, class_idx].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = (w * self.acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_cam(img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    import matplotlib.pyplot as plt
    import io as _io

    cam_uint8 = (cam * 255).astype(np.uint8)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(2, 2)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.set_axis_off()
    ax.imshow(cam_uint8, cmap="jet", interpolation="nearest")
    buf = _io.BytesIO()
    fig.canvas.print_png(buf)
    plt.close(fig)
    heat = Image.open(buf).convert("RGBA").resize(img.size, Image.BILINEAR)
    base = img.convert("RGBA")
    return Image.blend(base, heat, alpha=alpha).convert("RGB")


# ------------- App state -------------
MODEL = None
CLASSES = []
MODEL_TYPE = None
ARCH = None
TFM = transforms.Compose(
    [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
)

# Auto-load at startup
if DEFAULT_WEIGHTS.exists():
    MODEL, CLASSES, MODEL_TYPE, ARCH = load_model(DEFAULT_WEIGHTS)
    START_STATUS = f"✅ Auto-loaded {MODEL_TYPE} ({ARCH}) with {len(CLASSES)} classes from {DEFAULT_WEIGHTS}"
else:
    START_STATUS = f"❌ Weights not found: {DEFAULT_WEIGHTS}. Set a valid path below and Reload."


def reload_weights(path_str: str) -> str:
    global MODEL, CLASSES, MODEL_TYPE, ARCH
    p = Path(path_str)
    if not p.exists():
        return f"❌ Weights not found: {p}"
    MODEL, CLASSES, MODEL_TYPE, ARCH = load_model(p)
    return f"✅ Loaded {MODEL_TYPE} ({ARCH}) with {len(CLASSES)} classes from {p}"


def predict(image: Image.Image, do_gradcam: bool = True, save_cam: bool = True):
    if MODEL is None:
        return "Load weights first.", None
    im = image.convert("RGB")
    x = TFM(im).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(MODEL(x), dim=1).squeeze(0)
        conf, idx = torch.max(probs, dim=0)
    pred = CLASSES[int(idx)]
    msg = f"Prediction: {pred}  |  Confidence: {float(conf):.3f}"

    overlay_img = None
    if do_gradcam and MODEL_TYPE == "resnet":
        cam = GradCAM(MODEL, MODEL.layer4[-1])(x, int(idx))
        overlay_img = overlay_cam(im.resize((IMG_SIZE, IMG_SIZE)), cam)
        if save_cam:
            ts = time.strftime("%Y%m%d-%H%M%S")
            out_path = SAVE_DIR / f"cam_{pred}_{ts}.jpg"
            overlay_img.save(out_path)
            msg += f"\n💾 Grad-CAM saved: {out_path}"

    return msg, (overlay_img or im)


# ------------- Build Gradio UI -------------
with gr.Blocks(title="DermAI Web") as demo:
    gr.Markdown(
        "# DermAI — Dermatology (Research)\nUpload an image, get prediction, and optional Grad-CAM (auto-loads weights)."
    )

    status = gr.Markdown(START_STATUS)

    with gr.Row():
        w_in = gr.Textbox(value=str(DEFAULT_WEIGHTS), label="Weights .pth path")
        reload_btn = gr.Button("Reload Weights")
    reload_btn.click(fn=reload_weights, inputs=w_in, outputs=status)

    with gr.Row():
        img = gr.Image(label="Input image", type="pil")
        with gr.Column():
            gc = gr.Checkbox(value=True, label="Grad-CAM (ResNet only)")
            save = gr.Checkbox(value=True, label=f"Save Grad-CAM to {SAVE_DIR}")
            run = gr.Button("Predict")
            out_text = gr.Markdown()
            out_img = gr.Image(label="Output / Grad-CAM")

    run.click(fn=predict, inputs=[img, gc, save], outputs=[out_text, out_img])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
