import io

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms


def load_model(model_path="model/medicinal_plant_classifier.pth", device=None):
    """Load a trained model and its class names."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(model_path, map_location=device)

    class_names = checkpoint.get("class_names")
    if class_names is None:
        raise ValueError("Model file must contain 'class_names'.")

    model = models.mobilenet_v2(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, device


def preprocess_image(image: Image.Image):
    """Convert a PIL image to a normalized tensor."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


def image_bytes_to_pil(image_bytes: bytes) -> Image.Image:
    """Decode bytes into a PIL image using OpenCV."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def predict_from_bytes(model, class_names, device, image_bytes: bytes):
    image = image_bytes_to_pil(image_bytes)
    tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)

    return class_names[idx.item()], float(conf.item())


if __name__ == "__main__":
    # Example usage:
    # python inference.py path/to/image.jpg
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)

    model, classes, device = load_model()
    with open(sys.argv[1], "rb") as f:
        image_bytes = f.read()

    label, confidence = predict_from_bytes(model, classes, device, image_bytes)
    print(f"Predicted: {label} ({confidence:.3f})")
