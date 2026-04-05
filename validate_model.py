import gzip
import io
import torch
from PIL import Image
from torchvision import transforms
from models.resnet_model import get_model


def load_compressed_state_dict(path: str):
    with gzip.open(path, "rb") as f:
        buffer = io.BytesIO(f.read())
    return torch.load(buffer, map_location="cpu")


def validate_model(path: str, image_path: str):
    state_dict = load_compressed_state_dict(path)
    model = get_model(num_classes=4)
    model.load_state_dict(state_dict)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        print("Output shape:", outputs.shape)
        print("Predicted class:", outputs.argmax(dim=1).item())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate a compressed PyTorch model artifact.")
    parser.add_argument("--model", required=True, help="Path to the compressed model file.")
    parser.add_argument("--image", required=True, help="Path to a sample ECG image.")
    args = parser.parse_args()

    validate_model(args.model, args.image)
