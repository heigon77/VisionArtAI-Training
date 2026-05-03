"""
train_class.py
==============
Art-style classifier trained on the WikiArt dataset.

Architecture : MobileNetV3-Small (timm) — lightweight and fast, suited
               for deployment after ONNX export.
Dataset      : huggan/wikiart (HuggingFace Hub) — 27 art styles.
Loss         : CrossEntropy with label smoothing 0.1.
Optimiser    : AdamW + Cosine Annealing LR.

Outputs
-------
    models/wikiart_model_aug_only.pth   — best checkpoint
    results/training_metrics.csv        — per-epoch loss / accuracy
    results/confusion_matrix_seaborn.png
    results/class_training_curves_clean.png

Requirements
------------
    pip install torch torchvision timm datasets torch_snippets
    pip install pandas seaborn scikit-learn matplotlib
"""

import os
import time
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
import timm
from torch_snippets import Report
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS     = 40
LR         = 5e-5           # Conservative LR — MobileNetV3 is small; prevents overfitting
IMG_SIZE   = 224
SEED       = 42
SAVE_PATH  = "models/wikiart_model_aug_only.pth"

torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Class list — 27 WikiArt art styles (must match the dataset label order)
# ---------------------------------------------------------------------------
STYLES = [
    "Abstract Expressionism", "Action painting",    "Analytical Cubism",
    "Art Nouveau",            "Baroque",             "Color Field Painting",
    "Contemporary Realism",   "Cubism",              "Early Renaissance",
    "Expressionism",          "Fauvism",             "High Renaissance",
    "Impressionism",          "Mannerism (Late Renaissance)", "Minimalism",
    "Naive Art (Primitivism)","New Realism",         "Northern Renaissance",
    "Pointillism",            "Pop Art",             "Post Impressionism",
    "Realism",                "Rococo",              "Romanticism",
    "Symbolism",              "Synthetic Cubism",    "Ukiyo-e",
]
NUM_CLASSES = len(STYLES)   # 27

# ---------------------------------------------------------------------------
# Data transforms
# ---------------------------------------------------------------------------

# Training: aggressive augmentation to improve generalisation across painting styles
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandAugment(num_ops=2, magnitude=9),     # Random augmentation policy
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],         # ImageNet mean
                         [0.229, 0.224, 0.225]),        # ImageNet std
])

# Validation: deterministic — resize + normalise only
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class WikiArtDataset(torch.utils.data.Dataset):
    """
    Thin wrapper around a HuggingFace Dataset split.

    Each sample is expected to have:
        sample["image"]  — PIL Image
        sample["style"]  — integer class label (0-26)
    """

    def __init__(self, ds, tf):
        self.ds = ds
        self.tf = tf

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return self.tf(item["image"].convert("RGB")), item["style"]

# ---------------------------------------------------------------------------
# Load dataset and create splits
# ---------------------------------------------------------------------------
print("Loading dataset...")
ds = load_dataset("huggan/wikiart", split="train")

# 85 / 15 train-val split using a deterministic random permutation
n_val = int(0.15 * len(ds))
idx = torch.randperm(len(ds))
train_idx, val_idx = idx[n_val:], idx[:n_val]

train_ds = WikiArtDataset(ds.select(train_idx), train_tf)
val_ds   = WikiArtDataset(ds.select(val_idx),   val_tf)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=4)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = timm.create_model(
    "mobilenetv3_small_100",
    pretrained=True,            # Start from ImageNet weights
    num_classes=NUM_CLASSES,
    drop_rate=0.3,              # Dropout before classifier — regularises the small head
).to(DEVICE)

# Label smoothing penalises overconfident predictions, improving calibration
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# AdamW with decoupled weight decay works well with transformers and CNNs alike
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)

# Cosine annealing: LR smoothly decays from LR → 0 over EPOCHS iterations
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
log = Report(EPOCHS)

best_acc       = 0.0
start_time     = time.time()
train_loss_hist, val_loss_hist = [], []
train_acc_hist,  val_acc_hist  = [], []

for epoch in range(EPOCHS):

    # ── Training phase ──────────────────────────────────────────────
    model.train()
    trn_correct, trn_total, trn_loss_epoch = 0, 0, 0.0

    for bx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        pred = model(images)
        loss = criterion(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(pred, 1)
        trn_loss_epoch += loss.item()
        trn_correct    += (predicted == labels).sum().item()
        trn_total      += labels.size(0)
        acc             = trn_correct / trn_total * 100

        log.record(epoch + (bx + 1) / len(train_loader),
                   trn_loss=loss.item(), trn_acc=acc, end="\r")

    # ── Validation phase ────────────────────────────────────────────
    model.eval()
    val_correct, val_total, val_loss_epoch = 0, 0, 0.0

    with torch.no_grad():
        for bx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            pred = model(images)
            loss = criterion(pred, labels)

            _, predicted = torch.max(pred, 1)
            val_loss_epoch += loss.item()
            val_correct    += (predicted == labels).sum().item()
            val_total      += labels.size(0)
            acc             = val_correct / val_total * 100

            log.record(epoch + (bx + 1) / len(val_loader),
                       val_loss=loss.item(), val_acc=acc, end="\r")

    scheduler.step()
    log.report_avgs(epoch + 1)

    # Record history for plots
    train_loss_hist.append(trn_loss_epoch / len(train_loader))
    val_loss_hist.append(val_loss_epoch   / len(val_loader))
    train_acc_hist.append(trn_correct / trn_total * 100)
    val_acc_hist.append(val_correct   / val_total * 100)

    # ── Save best checkpoint ────────────────────────────────────────
    current_acc = val_correct / val_total * 100
    if current_acc > best_acc:
        # Cache the last batch predictions for the confusion matrix
        all_preds  = predicted.cpu().numpy()
        all_labels = labels.cpu().numpy()

        best_acc = current_acc
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "styles":               STYLES,
            "val_acc":              best_acc,
        }, SAVE_PATH)

elapsed = time.time() - start_time
print(f"\nTraining complete in {elapsed / 3600:.1f} h | Best val_acc: {best_acc:.2f}%")

# ---------------------------------------------------------------------------
# Save metrics CSV
# ---------------------------------------------------------------------------
os.makedirs("results", exist_ok=True)

df = pd.DataFrame({
    "epoch":      list(range(1, len(train_loss_hist) + 1)),
    "train_loss": train_loss_hist,
    "val_loss":   val_loss_hist,
    "train_acc":  train_acc_hist,
    "val_acc":    val_acc_hist,
})
df.to_csv("results/training_metrics.csv", index=False)

# ---------------------------------------------------------------------------
# Confusion matrix (last best-epoch batch — approximate)
# ---------------------------------------------------------------------------
cm      = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)  # Row-normalise

plt.figure(figsize=(12, 10))
sns.heatmap(cm_norm, cmap="Blues",
            xticklabels=STYLES, yticklabels=STYLES, square=True)
plt.title("Confusion Matrix (Normalised)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=90, fontsize=6)
plt.yticks(fontsize=6)
plt.tight_layout()
plt.savefig("results/confusion_matrix_seaborn.png", dpi=200)
plt.close()

# ---------------------------------------------------------------------------
# Training curves (with optional smoothing)
# ---------------------------------------------------------------------------

def smooth(y, window=5):
    """Simple moving average to reduce curve noise for visualisation."""
    if len(y) < window:
        return y
    return [
        sum(y[max(0, i - window):i + 1]) / len(y[max(0, i - window):i + 1])
        for i in range(len(y))
    ]

train_loss_s = smooth(train_loss_hist)
val_loss_s   = smooth(val_loss_hist)
train_acc_s  = smooth(train_acc_hist)
val_acc_s    = smooth(val_acc_hist)

epochs_x = range(1, len(train_loss_s) + 1)

plt.style.use("seaborn-v0_8-darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Loss curve
axes[0].plot(epochs_x, train_loss_s, label="Train",      linewidth=2)
axes[0].plot(epochs_x, val_loss_s,   label="Validation", linewidth=2, linestyle="--")
axes[0].set_title("Loss per Epoch")
axes[0].set_xlabel("Epochs")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Accuracy curve
axes[1].plot(epochs_x, train_acc_s, label="Train",      linewidth=2)
axes[1].plot(epochs_x, val_acc_s,   label="Validation", linewidth=2, linestyle="--")
axes[1].set_title("Accuracy per Epoch")
axes[1].set_xlabel("Epochs")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/class_training_curves_clean.png", dpi=200)
plt.close()
