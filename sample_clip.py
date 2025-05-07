import clip
import torch
from PIL import Image
import os

def load_clip():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cargar modelo CLIP
    model, preprocess = clip.load("ViT-B/32", device=device)
    model = model.encode_image  # Usar solo el encoder de imágenes
    
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
    model, preprocess, device = load_clip()
    
    # Ruta de ejemplo (debe ser reemplazada con la ruta real del dataset)
    image_path = os.path.join('simple1K', 'images', 'dolphin', '240002.jpg')
    
    if os.path.exists(image_path):
        features = extract_features(image_path, model, preprocess, device)
        print(f"Feature dimension: {features.shape[0]}")
    else:
        print(f"Example image not found at: {image_path}")
        print("Please provide a valid image path from your dataset")