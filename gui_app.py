# gui_app.py — Desktop GUI for DermAI (ResNet & SimpleCNN)
# - Load weights (.pth), open image, Predict, Grad-CAM (ResNet only)
# - Batch predict folder -> predictions.csv
# Run: python gui_app.py

import os, io, csv
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch, torch.nn as nn
from torchvision import models, transforms
import numpy as np

IMG_SIZE = 224
WINDOW_W, WINDOW_H = 1120, 720

# -------------------- Model helpers --------------------
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
        return self.classifier(self.features(x))

def detect_ckpt_type(ckpt):
    if "arch" in ckpt:
        a = ckpt["arch"].lower()
        return ("resnet", a if a in {"resnet18","resnet34","resnet50"} else "resnet18")
    keys = list(ckpt.get("model_state", {}).keys())
    if any(k.startswith("layer1.") or k.startswith("fc.") for k in keys):
        return ("resnet","resnet18")
    if any(k.startswith("features.0") or k.startswith("classifier.1") for k in keys):
        return ("cnn", None)
    return ("resnet","resnet18")

def load_model_from_ckpt(weights_path: Path):
    ckpt = torch.load(weights_path, map_location="cpu")
    classes = ckpt.get("classes")
    if classes is None:
        raise ValueError("Checkpoint missing 'classes'.")
    mtype, arch = detect_ckpt_type(ckpt)
    if mtype == "resnet":
        if arch == "resnet34":
            model = models.resnet34(weights=None)
        elif arch == "resnet50":
            model = models.resnet50(weights=None)
        else:
            arch = "resnet18"
            model = models.resnet18(weights=None)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, len(classes))
    else:
        model = SimpleCNN(num_classes=len(classes))
    missing = model.load_state_dict(ckpt["model_state"], strict=False)
    params_loaded = sum(p.numel() for n,p in model.named_parameters() if n not in missing.missing_keys)
    if params_loaded == 0 and missing.unexpected_keys:
        raise ValueError(
            "Checkpoint format does not match model.\n"
            "• If this is a ResNet checkpoint, use runs/resnet/best.pth.\n"
            "• If this is a CNN checkpoint, use runs/cnn/best.pth."
        )
    model.eval()
    return model, classes, mtype, (arch or "cnn")

# -------------------- Grad-CAM --------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.acts = None; self.grads = None
        self.h1 = target_layer.register_forward_hook(self._fwd)
        self.h2 = target_layer.register_full_backward_hook(self._bwd)
    def _fwd(self, m, i, o): self.acts = o.detach()
    def _bwd(self, m, gin, gout): self.grads = gout[0].detach()
    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None: class_idx = int(torch.argmax(logits,1).item())
        score = logits[:, class_idx]
        score.backward()
        w = self.grads.mean(dim=(2,3), keepdim=True)
        cam = (w * self.acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def overlay_cam(im_pil: Image.Image, cam: np.ndarray, alpha: float = 0.45):
    import matplotlib.pyplot as plt, io as _io
    cam_uint8 = (cam * 255).astype(np.uint8)
    fig = plt.figure(frameon=False); fig.set_size_inches(2,2)
    ax = plt.Axes(fig, [0,0,1,1]); fig.add_axes(ax); ax.set_axis_off()
    ax.imshow(cam_uint8, cmap='jet', interpolation='nearest')
    buf = _io.BytesIO(); fig.canvas.print_png(buf); plt.close(fig)
    heat = Image.open(buf).convert("RGBA").resize(im_pil.size, Image.BILINEAR)
    base = im_pil.convert("RGBA")
    return Image.blend(base, heat, alpha=alpha).convert("RGB")

# -------------------- GUI --------------------
class DermGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DermAI – GUI (ResNet & CNN)")
        self.geometry(f"{WINDOW_W}x{WINDOW_H}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None; self.classes = []
        self.model_type = None; self.arch = None
        self.cur_img = None

        self._build()

    def _build(self):
        bar = ttk.Frame(self); bar.pack(fill=tk.X, padx=10, pady=8)
        self.btn_load = ttk.Button(bar, text="Load Weights", command=self.on_load)
        self.btn_open = ttk.Button(bar, text="Open Image", command=self.on_open, state=tk.DISABLED)
        self.btn_pred = ttk.Button(bar, text="Predict", command=self.on_predict, state=tk.DISABLED)
        self.btn_cam  = ttk.Button(bar, text="Grad-CAM", command=self.on_cam, state=tk.DISABLED)
        self.btn_batch= ttk.Button(bar, text="Batch Folder Predict", command=self.on_batch, state=tk.DISABLED)
        self.btn_load.pack(side=tk.LEFT); self.btn_open.pack(side=tk.LEFT, padx=8)
        self.btn_pred.pack(side=tk.LEFT); self.btn_cam.pack(side=tk.LEFT, padx=8); self.btn_batch.pack(side=tk.LEFT)

        self.status = ttk.Label(self, text=f"Device: {self.device}")
        self.status.pack(fill=tk.X, padx=10)

        self.canvas = tk.Canvas(self, bg="#222", width=820, height=560)
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)

        right = ttk.Frame(self); right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        ttk.Label(right, text="Prediction").pack(anchor="w")
        self.pred_var = tk.StringVar(value="—"); ttk.Label(right, textvariable=self.pred_var, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(right, text="Confidence").pack(anchor="w", pady=(10,0))
        self.conf_var = tk.StringVar(value="—"); ttk.Label(right, textvariable=self.conf_var).pack(anchor="w")
        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(right, text="Classes").pack(anchor="w")
        self.lst = tk.Listbox(right, height=14); self.lst.pack(fill=tk.BOTH, expand=True)

    # ---------- actions ----------
    def on_load(self):
        p = filedialog.askopenfilename(title="Select weights (.pth)", filetypes=[("PyTorch checkpoint","*.pth")])
        if not p: return
        try:
            model, classes, mtype, arch = load_model_from_ckpt(Path(p))
            self.model, self.classes, self.model_type, self.arch = model.to(self.device), classes, mtype, arch
        except Exception as e:
            messagebox.showerror("Load error", str(e)); return
        self.status.config(text=f"Loaded {self.model_type} ({self.arch}) • {len(self.classes)} classes • Device: {self.device}")
        self.btn_open.config(state=tk.NORMAL); self.btn_pred.config(state=tk.NORMAL); self.btn_batch.config(state=tk.NORMAL)
        self.btn_cam.config(state=(tk.NORMAL if self.model_type=="resnet" else tk.DISABLED))
        self.lst.delete(0, tk.END); [self.lst.insert(tk.END, c) for c in self.classes]

    def on_open(self):
        p = filedialog.askopenfilename(title="Open image", filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")])
        if not p: return
        try: self.cur_img = Image.open(p).convert("RGB")
        except Exception as e: messagebox.showerror("Image error", str(e)); return
        self._show(self.cur_img); self.pred_var.set("—"); self.conf_var.set("—")

    def _show(self, im: Image.Image):
        cw, ch = int(self.canvas['width']), int(self.canvas['height'])
        w, h = im.size; s = min(cw/w, ch/h)
        disp = im.resize((max(1,int(w*s)), max(1,int(h*s))), Image.LANCZOS)
        self._tk = ImageTk.PhotoImage(disp)
        self.canvas.delete("all"); self.canvas.create_image(cw//2, ch//2, image=self._tk)

    def on_predict(self):
        if self.model is None or self.cur_img is None:
            messagebox.showinfo("Info", "Load weights and open an image first."); return
        tfm = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
        x = tfm(self.cur_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            conf, idx = torch.max(probs, dim=0)
        pred = self.classes[int(idx.item())]
        self.pred_var.set(pred)
        self.conf_var.set(f"{float(conf.item()):.3f}")

    def on_cam(self):
        if self.model is None or self.cur_img is None:
            messagebox.showinfo("Info", "Load weights and open an image first."); return
        if not hasattr(self.model, 'layer4'):
            messagebox.showwarning("Grad-CAM", "Grad-CAM is supported for ResNet models only."); return
        tfm = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
        x = tfm(self.cur_img).unsqueeze(0).to(self.device)
        with torch.no_grad(): idx = int(torch.argmax(self.model(x), dim=1).item())
        cam_engine = GradCAM(self.model, self.model.layer4[-1])
        cam = cam_engine(x, class_idx=idx)
        overlay = overlay_cam(self.cur_img.resize((IMG_SIZE, IMG_SIZE)), cam)
        self._show(overlay)

    def on_batch(self):
        if self.model is None:
            messagebox.showinfo("Info", "Load weights first."); return
        folder = filedialog.askdirectory(title="Select folder with images")
        if not folder: return
        exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff")
        paths = []
        for e in exts: paths += list(Path(folder).rglob(e))
        if not paths: messagebox.showinfo("Info", "No images found."); return
        tfm = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
        rows = []
        for p in paths:
            try:
                im = Image.open(p).convert("RGB")
                x = tfm(im).unsqueeze(0)
                with torch.no_grad():
                    logits = self.model(x)
                    probs = torch.softmax(logits, dim=1).squeeze(0)
                    conf, idx = torch.max(probs, dim=0)
                rows.append({"image": str(p), "pred_class": self.classes[int(idx)], "confidence": float(conf)})
            except Exception as e:
                rows.append({"image": str(p), "pred_class": "", "confidence": "", "error": str(e)})
        out = Path(folder)/"predictions.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            fn = ["image","pred_class","confidence","error"]
            w = csv.DictWriter(f, fieldnames=fn); w.writeheader(); [w.writerow(r) for r in rows]
        messagebox.showinfo("Done", f"Saved {len(rows)} predictions to\n{out}")

if __name__ == "__main__":
    DermGUI().mainloop()
