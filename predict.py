import torch
from PIL import Image
from torchvision import transforms
from models.resnet_model import get_model


def predict(image_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(num_classes=4)
    model.load_state_dict(torch.load("best_model.pth"))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1).squeeze()
        _, predicted = torch.max(outputs, 1)

    class_names = ["abnormal", "history_mi", "mi", "normal"]
    pred_class = class_names[predicted.item()]
    confidence = probs[predicted.item()].item() * 100

    print(f"\n{'=' * 40}")
    print(f"  Prediction : {pred_class}")
    print(f"  Confidence : {confidence:.2f}%")
    print(f"{'=' * 40}")
    print(f"\n  Class Probabilities:")
    for i, name in enumerate(class_names):
        bar = "█" * int(probs[i].item() * 30)
        print(f"    {name:>12s}: {probs[i].item()*100:6.2f}%  {bar}")
    print()


import os

if __name__ == "__main__":
    path = input("Enter image path: ").strip()
    
    # Remove accidental quotes
    path = path.replace('"', '').replace("'", "")
    
    # Normalize Windows path
    path = os.path.normpath(path)

    predict(path)
