# Dermatology Preliminary Diagnosis Tool

This package provides a ready-to-run pipeline for **preliminary** (non-clinical) dermatology image diagnosis using:
- A custom **CNN** classifier
- A **ResNet** (transfer learning) classifier
- A **YOLOv5** detector (optional) for lesion localization

> ⚠️ Medical disclaimer: This software is for research/education only and **not** a medical device.

## Dataset
- Auto-detected 903 images from the provided archive.
- Labels are inferred from the immediate parent folder names.
- Detected classes: Atopic Dermatitis, Tinea Ringworm Candidiasis, Actinic keratosis, Benign keratosis, Dermatofibroma, Melanocytic nevus, Melanoma, Squamous cell carcinoma, Vascular lesion.

Your extracted dataset is at:
```
/mnt/data/derm_dataset
```

### Expected Layout
```
dataset_root/
  class_A/
    img1.jpg
    img2.jpg
  class_B/
    ...
```

## Quickstart

1) Create a virtual environment and install requirements:
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2) Split the dataset (train/val/test):
```bash
python tools/split_dataset.py --root "/mnt/data/derm_dataset" --out data_splits --train 0.7 --val 0.2 --test 0.1
```

3) Train models:

Custom CNN:
```bash
python train_cnn.py --data data_splits --epochs 15 --batch-size 32 --img-size 224
```

ResNet (Transfer Learning):
```bash
python train_resnet.py --data data_splits --epochs 10 --batch-size 32 --img-size 224 --lr 3e-4
```

YOLOv5 (optional; needs detection labels or auto box):
```bash
python train_yolov5.py --data data_splits --epochs 50
```

4) Inference on a folder of images:
```bash
python infer.py --weights runs/resnet/best.pth --arch resnet18 --input sample_inputs --out outputs
```

## Notes
- If your dataset does **not** have bounding boxes, `train_yolov5.py` can generate a pseudo box covering most of the image (center crop rectangle). Replace with real annotations when available.
- All scripts log to `runs/`.

## Grad-CAM (Model Explainability)

Generate **Grad-CAM heatmaps** to visualize which regions influenced the ResNet prediction:

```bash
# After training ResNet (weights at runs/resnet/best.pth)
python gradcam.py --weights runs/resnet/best.pth --arch resnet18 --input sample_inputs --out outputs/gradcam
```

- Overlays are saved to `outputs/gradcam/cam/`.
- A `results.json` summarizing predictions and file paths is also created.
```

