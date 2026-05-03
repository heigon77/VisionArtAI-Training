"""
export_class_onnx.py
====================
Exports the trained WikiArt style classifier to ONNX and then applies
dynamic INT8 quantisation to reduce model size for edge / CPU deployment.

Outputs
-------
    style_classifier.onnx       — standard ONNX model (FP32 weights)
    style_classifier_int8.onnx  — INT8 quantised model (~4× smaller)

Requirements
------------
    pip install torch timm onnx onnxruntime

Usage
-----
    python export_class_onnx.py
"""

import torch
import timm
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
CHECKPOINT = Path("models/wikiart_model_aug_only.pth")  # Saved by train_class.py
ONNX_PATH  = Path("style_classifier.onnx")             # FP32 export destination
INT8_PATH  = Path("style_classifier_int8.onnx")        # INT8 export destination
IMG_SIZE   = 224                                        # Must match training resolution

# ---------------------------------------------------------------------------
# Reload model from checkpoint
# ---------------------------------------------------------------------------
ckpt = torch.load(CHECKPOINT, map_location="cpu")

# Recreate the exact same architecture used during training
model = timm.create_model(
    "mobilenetv3_small_100",
    pretrained=False,                       # Weights come from our checkpoint, not ImageNet
    num_classes=len(ckpt["styles"]),        # 27 WikiArt styles
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()   # Disable dropout and BatchNorm running stats tracking

# ---------------------------------------------------------------------------
# Export to ONNX (FP32)
# ---------------------------------------------------------------------------
# Dummy input — the exporter traces the graph with this tensor
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

torch.onnx.export(
    model,
    dummy,
    ONNX_PATH,
    opset_version=17,                       # Opset 17 has broad runtime support (ORT ≥ 1.14)
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={
        "image":  {0: "batch"},             # Allow variable batch size at inference time
        "logits": {0: "batch"},
    },
)
print(f"ONNX exported: {ONNX_PATH}")

# ---------------------------------------------------------------------------
# Dynamic INT8 quantisation
# ---------------------------------------------------------------------------
# quantize_dynamic replaces weight tensors with INT8 representations.
# Activations are still computed in FP32 at runtime, so no calibration
# dataset is required — unlike static quantisation.
# Trade-off: ~4× size reduction, ~1-2 pp accuracy drop on average.
quantize_dynamic(
    model_input=str(ONNX_PATH),
    model_output=str(INT8_PATH),
    weight_type=QuantType.QInt8,
)
print(f"INT8 exported: {INT8_PATH}")

# ---------------------------------------------------------------------------
# File size comparison
# ---------------------------------------------------------------------------
print(f"ONNX size : {ONNX_PATH.stat().st_size / 1e6:.1f} MB")
print(f"INT8 size : {INT8_PATH.stat().st_size / 1e6:.1f} MB")
