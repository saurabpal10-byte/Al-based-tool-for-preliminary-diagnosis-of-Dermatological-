# pretrained_loader.py
import torch
import torch.nn as nn
from torchvision import models

# --- Build a strong pretrained classifier head on top of a solid backbone ---
# Supported: resnet18/34/50, convnext_base, efficientnet_v2_m
# Returns: model, target_layer (for Grad-CAM), arch_name

def _replace_classifier_convnext(model, num_classes):
    in_feats = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_feats, num_classes)
    return model

def _replace_classifier_efficientnet_v2_m(model, num_classes):
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, num_classes)
    return model

def _replace_classifier_resnet(model, num_classes):
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model

def build_pretrained(backbone: str, num_classes: int):
    backbone = backbone.lower()

    if backbone == "convnext_base":
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        model = _replace_classifier_convnext(model, num_classes)
        # last ConvNeXt block (good Grad-CAM target)
        target_layer = model.features[-1][-1].block[2] if hasattr(model.features[-1][-1], "block") else model.features[-1][-1]
        arch = "convnext_base"

    elif backbone == "efficientnet_v2_m":
        model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        model = _replace_classifier_efficientnet_v2_m(model, num_classes)
        # last features stage contains conv blocks; good Grad-CAM target
        target_layer = model.features[-1]  # fused MBConv stack (has convs)
        arch = "efficientnet_v2_m"

    elif backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model = _replace_classifier_resnet(model, num_classes)
        target_layer = model.layer4[-1]
        arch = "resnet50"

    elif backbone == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model = _replace_classifier_resnet(model, num_classes)
        target_layer = model.layer4[-1]
        arch = "resnet34"

    else:
        # default: resnet18
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model = _replace_classifier_resnet(model, num_classes)
        target_layer = model.layer4[-1]
        arch = "resnet18"

    return model, target_layer, arch


# ----- Loader helpers that match your checkpoint format -----

def init_for_training(backbone: str, classes):
    model, target_layer, arch = build_pretrained(backbone, num_classes=len(classes))
    return model, target_layer, arch

def load_from_checkpoint(ckpt_path: str):
    """Loads a model from your saved checkpoint format: dict(model_state, classes, arch)."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt["classes"]
    arch = ckpt.get("arch", "resnet18")
    model, target_layer, arch = build_pretrained(arch, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, target_layer, classes, arch
