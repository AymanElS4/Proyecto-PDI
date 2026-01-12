import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 3)

    model1 = torch.load(
        "identf/resnet18_arboles.pth",
        map_location=device
    )
    model.load_state_dict(model1["model_state"])
    model.to(device)
    model.eval()

    classes = model1["classes"]

    return model, classes, device

#cambiar, es mas, creo que se lo deberia eliminar
def predict_top2(image_file, model, classes, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_file).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        k = min(2, len(classes))
        top_probs, top_idxs = torch.topk(probs, k)

    return [
        {
            "especie": classes[top_idxs[0][i].item()],
            "confianza": float(top_probs[0][i] * 100)
        }
        for i in range(2)
    ]