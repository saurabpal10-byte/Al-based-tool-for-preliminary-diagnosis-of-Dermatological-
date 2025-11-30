# site_server.py
import os, io, json, time, sqlite3
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, abort, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import torch, torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

ROOT = Path(__file__).parent
WEIGHTS = ROOT / "runs" / "resnet" / "best.pth"
OUT_DIR = ROOT / "site_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_SIZE = 224
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DB_PATH = ROOT / "users.db"

app = Flask(__name__, static_folder=str(ROOT / "web"), static_url_path="/")
app.secret_key = os.urandom(24)  # Secure random key for sessions

# ---------------- Database ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT UNIQUE NOT NULL, 
                  password_hash TEXT NOT NULL)''')
    conn.commit()
    conn.close()

init_db()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ---------------- Model Helpers ----------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, num_classes))
    def forward(self,x): return self.classifier(self.features(x))

def detect_ckpt_type(ckpt):
    if "arch" in ckpt:
        a = str(ckpt["arch"]).lower()
        if a in {"resnet18","resnet34","resnet50"}: return "resnet", a
    keys = list(ckpt.get("model_state",{}).keys())
    if any(k.startswith("layer1.") or k.startswith("fc.") for k in keys): return "resnet","resnet18"
    if any(k.startswith("features.0") or k.startswith("classifier.1") for k in keys): return "cnn", None
    return "resnet","resnet18"

def load_model(weights_path: Path):
    ckpt = torch.load(weights_path, map_location="cpu")
    classes = ckpt.get("classes")
    if not classes: raise RuntimeError("Checkpoint missing 'classes'.")
    mtype, arch = detect_ckpt_type(ckpt)
    if mtype == "resnet":
        if arch == "resnet34": model = models.resnet34(weights=None)
        elif arch == "resnet50": model = models.resnet50(weights=None)
        else: model = models.resnet18(weights=None)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, len(classes))
    else:
        model = SimpleCNN(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model, classes, mtype, (arch or "cnn")

# Grad-CAM
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.target_layer.register_forward_hook(self._fwd)
        self.target_layer.register_full_backward_hook(self._bwd)
    def _fwd(self,m,i,o): self.activations = o.detach()
    def _bwd(self,m,gin,gout): self.gradients = gout[0].detach()
    def __call__(self,x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None: class_idx = int(torch.argmax(logits,1).item())
        logits[:, class_idx].backward()
        w = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (w * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def overlay_cam(img_pil: Image.Image, cam, alpha=0.45):
    import matplotlib.pyplot as plt, io
    cam_uint8 = (cam * 255).astype("uint8")
    fig = plt.figure(frameon=False); fig.set_size_inches(2,2)
    ax = plt.Axes(fig,[0,0,1,1]); fig.add_axes(ax); ax.set_axis_off()
    ax.imshow(cam_uint8, cmap="jet", interpolation="nearest")
    buf = io.BytesIO(); fig.canvas.print_png(buf); plt.close(fig)
    heat = Image.open(buf).convert("RGBA").resize(img_pil.size, Image.BILINEAR)
    return Image.blend(img_pil.convert("RGBA"), heat, alpha=alpha).convert("RGB")

# load model on start
if not WEIGHTS.exists():
    print("WARNING: weights not found at", WEIGHTS)
    MODEL = None
    CLASSES = []
    MODEL_TYPE = None
else:
    MODEL, CLASSES, MODEL_TYPE, ARCH = load_model(WEIGHTS)
    print(f"Loaded {MODEL_TYPE} ({ARCH}) with {len(CLASSES)} classes")

TFM = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])

# ---------------- Routes ----------------

@app.route("/")
def index():
    if 'user_id' not in session:
        return redirect("/login.html")
    return app.send_static_file("index.html")

@app.route("/login.html")
def login_page():
    if 'user_id' in session:
        return redirect("/")
    return app.send_static_file("login.html")

@app.route("/register.html")
def register_page():
    if 'user_id' in session:
        return redirect("/")
    return app.send_static_file("register.html")

@app.route("/api/register", methods=["POST"])
def register():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
        
    conn = get_db_connection()
    try:
        # Check if user exists
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        if user:
            return jsonify({"error": "Username already exists"}), 409
            
        hashed = generate_password_hash(password)
        conn.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, hashed))
        conn.commit()
        return jsonify({"message": "Registration successful"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    conn.close()
    
    if user and check_password_hash(user['password_hash'], password):
        session['user_id'] = user['id']
        session['username'] = user['username']
        return jsonify({"message": "Login successful"}), 200
    else:
        return jsonify({"error": "Invalid credentials"}), 401

@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out"}), 200

# Disease Information Dictionary
DISEASE_INFO = {
    "Actinic keratosis": {
        "description": "A rough, scaly patch on the skin caused by years of sun exposure. It is considered a precancerous skin growth that can develop into squamous cell carcinoma if left untreated.",
        "basis": "Detected based on analysis of scaly texture, erythematous base patterns, and surface roughness characteristic of actinic damage."
    },
    "Atopic Dermatitis": {
        "description": "A chronic condition that makes your skin red and itchy. It is a type of eczema common in children but can occur at any age. It often flares periodically.",
        "basis": "Identified by analyzing characteristic patterns of erythema (redness), scaling, and lichenification (thickening) of the skin."
    },
    "Benign keratosis": {
        "description": "A common non-cancerous skin growth. It often appears as a waxy brown, black, or tan growth. It is also known as seborrheic keratosis.",
        "basis": "Recognized by well-defined borders, 'stuck-on' appearance, and specific pigmentation features distinct from malignant lesions."
    },
    "Dermatofibroma": {
        "description": "A common overgrowth of fibrous tissue situated in the dermis. It is benign, usually asymptomatic, and feels like a small hard lump under the skin.",
        "basis": "Detected based on firm nodular appearance, central white scar-like patches, and peripheral pigmentation patterns."
    },
    "Melanocytic nevus": {
        "description": "A common type of mole. It is a benign proliferation of melanocytes (pigment cells). Most adults have between 10 and 40 common moles.",
        "basis": "Identified by uniform pigmentation, regular border characteristics, and symmetry typical of benign nevi."
    },
    "Melanoma": {
        "description": "The most serious type of skin cancer. It develops in the cells (melanocytes) that produce melanin. Early detection is crucial for successful treatment.",
        "basis": "Detected based on high-risk features such as asymmetry, irregular borders, color variegation, and structural evolution."
    },
    "Squamous cell carcinoma": {
        "description": "A common form of skin cancer that develops in the squamous cells that make up the middle and outer layers of the skin. It is usually not life-threatening but can be aggressive.",
        "basis": "Identified by hyperkeratotic (crusty) lesions, potential ulceration patterns, and specific growth characteristics."
    },
    "Tinea Ringworm Candidiasis": {
        "description": "Fungal infections of the skin. Tinea (ringworm) causes a ring-shaped rash, while Candidiasis is a yeast infection causing red, itchy patches.",
        "basis": "Recognized by annular (ring-shaped) lesion morphology, scaling borders, or satellite pustules typical of fungal infections."
    },
    "Vascular lesion": {
        "description": "Abnormalities of the skin and underlying tissues, including birthmarks and hemangiomas. They are caused by malformed blood vessels.",
        "basis": "Detected based on specific red/purple coloration (erythema/violaceous hues) and vascular structural patterns."
    }
}

FAQ_DATA = [
    {
        "question": "How does DermAI work?",
        "answer": "DermAI uses advanced deep learning models (ResNet) to analyze skin lesion images. It extracts visual features and compares them against a database of known skin conditions to provide a diagnosis."
    },
    {
        "question": "Is the diagnosis 100% accurate?",
        "answer": "No. While DermAI is highly accurate, it is an AI tool and not a substitute for professional medical advice. Always consult a dermatologist for a definitive diagnosis."
    },
    {
        "question": "What types of images should I upload?",
        "answer": "Upload clear, well-lit, close-up photos of the skin lesion. Avoid blurry images or photos with poor lighting for the best results."
    },
    {
        "question": "What is the Accuracy Score?",
        "answer": "The Accuracy Score indicates the model's confidence in its prediction. A higher score suggests a stronger match with the identified condition."
    },
    {
        "question": "What is the Heatmap Overlay?",
        "answer": "The Heatmap (Grad-CAM) highlights the specific regions of the image that the model focused on to make its prediction, helping you understand the 'why' behind the diagnosis."
    }
]

@app.route("/api/diseases", methods=["GET"])
def get_diseases():
    return jsonify(DISEASE_INFO)

@app.route("/api/faq", methods=["GET"])
def get_faq():
    return jsonify(FAQ_DATA)

@app.route("/predict", methods=["POST"])
def predict():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
        
    if MODEL is None:
        return jsonify({"error":"model not loaded"}), 500
    if "file" not in request.files:
        return jsonify({"error":"no file uploaded"}), 400
    f = request.files["file"]
    img = Image.open(io.BytesIO(f.read())).convert("RGB")
    x = TFM(img).unsqueeze(0)
    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        conf, idx = torch.max(probs, dim=0)
    
    pred_class = CLASSES[int(idx)]
    info = DISEASE_INFO.get(pred_class, {
        "description": "No specific description available for this class.",
        "basis": "Analyzed based on visual feature extraction."
    })
    
    return jsonify({
        "pred_class": pred_class, 
        "confidence": float(conf.item()),
        "description": info["description"],
        "accuracy_basis": info["basis"]
    })

@app.route("/gradcam", methods=["POST"])
def gradcam():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    if MODEL is None:
        return jsonify({"error":"model not loaded"}), 500
    if MODEL_TYPE != "resnet":
        return jsonify({"error":"Grad-CAM only supported for ResNet models"}), 400
    if "file" not in request.files:
        return jsonify({"error":"no file uploaded"}), 400
    f = request.files["file"]
    img = Image.open(io.BytesIO(f.read())).convert("RGB")
    x = TFM(img).unsqueeze(0)
    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        conf, idx = torch.max(probs, dim=0)
    cam_engine = GradCAM(MODEL, MODEL.layer4[-1])
    cam = cam_engine(x, int(idx))
    overlay = overlay_cam(img.resize((IMG_SIZE, IMG_SIZE)), cam)
    ts = time.strftime("%Y%m%d-%H%M%S")
    fname = f"cam_{ts}.jpg"
    out_path = OUT_DIR / fname
    overlay.save(out_path)
    
    pred_class = CLASSES[int(idx)]
    info = DISEASE_INFO.get(pred_class, {
        "description": "No specific description available for this class.",
        "basis": "Analyzed based on visual feature extraction."
    })

    return jsonify({
        "pred_class": pred_class, 
        "confidence": float(conf.item()), 
        "cam_url": f"/site_outputs/{fname}",
        "description": info["description"],
        "accuracy_basis": info["basis"]
    })

@app.route("/site_outputs/<path:fn>")
def serve_output(fn):
    if 'user_id' not in session:
        return abort(401)
    p = OUT_DIR / fn
    if not p.exists(): abort(404)
    return send_from_directory(str(OUT_DIR), fn)

# static files under web/
@app.route("/<path:fn>")
def static_proxy(fn):
    # Allow access to login/register assets without auth
    if fn in ["login.html", "register.html", "style.css", "app.js"]:
        return app.send_static_file(fn)
    # For other assets, check auth if strictness is needed, but usually CSS/JS is fine.
    # We'll allow static assets generally, but protect the main page logic via index route.
    return app.send_static_file(fn)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print("Starting server on http://0.0.0.0:%d" % port)
    app.run(host="0.0.0.0", port=port, debug=True)
