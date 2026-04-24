import gzip
import io
import os

import torch
from PIL import Image
from torchvision import transforms
from models.resnet_model import get_model

MODEL_PATH = "best_model_state_dict.pth.gz"
CLASS_NAMES = ["abnormal", "history_mi", "mi", "normal"]


def load_compressed_state_dict(path: str):
    with gzip.open(path, "rb") as f:
        buffer = io.BytesIO(f.read())
    return torch.load(buffer, map_location="cpu")


def predict(image_path: str, model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Compressed model not found at {model_path}")

    state_dict = load_compressed_state_dict(model_path)
    model = get_model(num_classes=4)
    model.load_state_dict(state_dict)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze()
        predicted_index = int(probs.argmax().item())

    pred_class = CLASS_NAMES[predicted_index]
    confidence = float(probs[predicted_index].item() * 100)

    print(f"\n{'=' * 40}")
    print(f"  Prediction : {pred_class}")
    print(f"  Confidence : {confidence:.2f}%")
    print(f"{'=' * 40}")
    print(f"\n  Class Probabilities:")
    for name, score in zip(CLASS_NAMES, probs.tolist()):
        bar = "█" * int(score * 30)
        print(f"    {name:>12s}: {score * 100:6.2f}%  {bar}")
    print()


if __name__ == "__main__":
    path = input("Enter image path: ").strip()
    path = path.replace('"', '').replace("'", "")
    path = os.path.normpath(path)
    predict(path)
