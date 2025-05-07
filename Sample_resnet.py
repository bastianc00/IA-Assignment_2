import torch
from torchvision import transforms, models
from PIL import Image
import os

def load_resnet(model_name='resnet34'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cargar modelo
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True).to(device)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=True).to(device)
    model.fc = torch.nn.Identity()  # Eliminar capa de clasificación
    
    # Preprocesamiento estándar para ResNet
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return model, preprocess, device

def extract_features(image_path, model, preprocess, device):
    # Cargar imagen desde la ruta completa
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model(image)
    
    return features.cpu().numpy().flatten()

if __name__ == '__main__':
    # Ejemplo de uso
    model, preprocess, device = load_resnet('resnet34')
    
    # Ruta de ejemplo (debe ser reemplazada con la ruta real del dataset)
    image_path = os.path.join('simple1K', 'images', 'dolphin', '240002.jpg')
    
    if os.path.exists(image_path):
        features = extract_features(image_path, model, preprocess, device)
        print(f"Feature dimension: {features.shape[0]}")
    else:
        print(f"Example image not found at: {image_path}")
        print("Please provide a valid image path from your dataset")