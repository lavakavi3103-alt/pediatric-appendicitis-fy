
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

MODEL_PATH = OUT / "cnn_roi_model.pth"
IMAGE_DIR = ROOT / "data/processed"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=== GRAD-CAM EXPLAINABILITY ===")
print("Device:", DEVICE)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

# Load saved checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

if isinstance(checkpoint, torch.nn.Module):
    model = checkpoint

else:
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format.")

    state = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict", checkpoint)
    )

    if not isinstance(state, dict):
        raise ValueError("Cannot find model weights.")

    # Remove DataParallel prefix if present
    state = {
        k.removeprefix("module."): v
        for k, v in state.items()
    }

    # This loader supports ResNet-18 checkpoints
    # with 1 or 2 output classes.
    if "fc.weight" not in state:
        raise ValueError(
            "Checkpoint is not a standard ResNet-18. "
            "Share your 09h_cnn_roi_train.py model "
            "definition so Grad-CAM can match it."
        )

    num_classes = state["fc.weight"].shape[0]

    if num_classes not in (1, 2):
        raise ValueError(
            f"Unexpected output classes: {num_classes}"
        )

    model = models.resnet18(weights=None)

    if state["conv1.weight"].shape[1] == 1:
        model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=7,
            stride=2, padding=3, bias=False
        )

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        num_classes
    )

    model.load_state_dict(state, strict=True)

model = model.to(DEVICE)
model.eval()

# Target layer: last convolutional layer of ResNet
if not hasattr(model, "layer4"):
    raise ValueError(
        "Model has no layer4. Custom CNN architecture "
        "needs its own Grad-CAM target layer."
    )

target_layer = model.layer4[-1]

# Find a sample ultrasound image
image_paths = []
for folder in [
    ROOT / "data/processed/roi_images",
    ROOT / "data/processed/images_roi",
    ROOT / "data/raw/images/US_Pictures",
]:
    if folder.exists():
        image_paths = sorted(
            p for p in folder.rglob("*")
            if p.suffix.lower() in {
                ".png", ".jpg", ".jpeg"
            }
        )
        if image_paths:
            break

if not image_paths:
    raise FileNotFoundError(
        "No ultrasound images found. "
        "Check your ROI image folder path."
    )

image_path = image_paths[0]
print("Image:", image_path)

image = Image.open(image_path).convert("RGB")

# IMPORTANT: Match training preprocessing here.
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

if model.conv1.in_channels == 1:
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])

input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

activations = []
gradients = []

def forward_hook(module, inputs, output):
    activations.append(output.detach())

    def save_gradient(grad):
        gradients.append(grad.detach())

    output.register_hook(save_gradient)

handle = target_layer.register_forward_hook(forward_hook)

try:
    with torch.enable_grad():
        output = model(input_tensor)

        if output.ndim == 1:
            output = output.unsqueeze(0)

        if output.shape[1] == 1:
            probability = torch.sigmoid(output)[0, 0]
            predicted_class = int(probability >= 0.5)
            score = output[0, 0]
            if predicted_class == 0:
                score = -score
        else:
            probabilities = torch.softmax(output, dim=1)
            predicted_class = int(
                probabilities.argmax(dim=1).item()
            )
            probability = probabilities[
                0, predicted_class
            ]
            score = output[0, predicted_class]

        model.zero_grad(set_to_none=True)
        score.backward()

    if not activations or not gradients:
        raise RuntimeError(
            "Grad-CAM hooks did not capture tensors."
        )

    activation = activations[-1]
    gradient = gradients[-1]

    weights = gradient.mean(dim=(2, 3), keepdim=True)

    cam = (
        weights * activation
    ).sum(dim=1, keepdim=True)

    cam = F.relu(cam)

    cam = F.interpolate(
        cam,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    cam = cam[0, 0].cpu().numpy()

    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

finally:
    handle.remove()

# Visualization
original = np.asarray(
    image.resize((224, 224))
) / 255.0

heatmap = plt.get_cmap("jet")(cam)[..., :3]
overlay = 0.6 * original + 0.4 * heatmap
overlay = np.clip(overlay, 0, 1)

fig, axes = plt.subplots(
    1, 3, figsize=(12, 4)
)

axes[0].imshow(original)
axes[0].set_title("Ultrasound Image")

axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
axes[1].set_title("Grad-CAM Heatmap")

axes[2].imshow(overlay)
axes[2].set_title("Grad-CAM Overlay")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

output_file = OUT / "gradcam_result.png"
plt.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight"
)
plt.close()

print("Predicted class index:", predicted_class)
print("Predicted class probability:", round(
    float(probability.item()), 4
))
print("Saved:", output_file)
print("Grad-CAM completed successfully.")
print(
    "NOTE: Verify class mapping and preprocessing "
    "against the original training script."
)
