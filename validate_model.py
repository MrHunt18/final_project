import argparse
import gzip
import io
import torch
from PIL import Image
from torchvision import transforms
from models.resnet_model import get_model


def load_compressed_state_dict(path: str):
    with gzip.open(path, "rb") as f:
        buffer = io.BytesIO(f.read())
    state_dict = torch.load(buffer, map_location="cpu")
    return state_dict


def test_model(compressed_model_path: str, image_path: str):
    state_dict = load_compressed_state_dict(compressed_model_path)
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
        output = model(input_tensor)
        predicted = output.argmax(dim=1).item()

    print("Model loaded successfully.")
    print("Output shape:", output.shape)
    print("Predicted class index:", predicted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a compressed PyTorch ECG model")
    parser.add_argument("--model", default="best_model_state_dict.pth.gz", help="Compressed model path")
    parser.add_argument("--image", default=None, help="Path to an ECG image for sanity check")
    args = parser.parse_args()

    if args.image is None:
        raise SystemExit("Please provide --image to validate the model with a sample image.")

    test_model(args.model, args.image)
