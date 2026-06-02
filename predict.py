import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import json
import argparse
import os

DISEASE_TR = {
    "Pepper__bell___Bacterial_spot": "Biber - Bakteriyel Leke",
    "Pepper__bell___healthy": "Biber - Sağlıklı",
    "Potato___Early_blight": "Patates - Erken Yanıklık",
    "Potato___Late_blight": "Patates - Geç Yanıklık",
    "Potato___healthy": "Patates - Sağlıklı",
    "Tomato_Bacterial_spot": "Domates - Bakteriyel Leke",
    "Tomato_Early_blight": "Domates - Erken Yanıklık",
    "Tomato_Late_blight": "Domates - Geç Yanıklık",
    "Tomato_Leaf_Mold": "Domates - Yaprak Küfü",
    "Tomato_Septoria_leaf_spot": "Domates - Septoria Yaprak Lekesi",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Domates - Kırmızı Örümcek",
    "Tomato__Target_Spot": "Domates - Hedef Leke",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Domates - Sarı Yaprak Kıvrılma Virüsü",
    "Tomato__Tomato_mosaic_virus": "Domates - Mozaik Virüsü",
    "Tomato_healthy": "Domates - Sağlıklı",
}

def load_model(model_path="model.pth", class_names_path="class_names.json"):
    with open(class_names_path) as f:
        class_names = json.load(f)
    model = models.efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, class_names

def predict_image(image_path, model, class_names):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = tf(image).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        prob = torch.softmax(out, 1)
        conf, pred = prob.max(1)
    name = class_names[pred.item()]
    disease_tr = DISEASE_TR.get(name, name)
    return name, disease_tr, conf.item()

def severity_analysis(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([25, 40, 40]), np.array([95, 255, 255]))
    d1 = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([25, 255, 200]))
    d2 = cv2.inRange(hsv, np.array([95, 30, 30]), np.array([180, 255, 200]))
    disease_mask = cv2.bitwise_or(d1, d2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel)
    leaf = cv2.bitwise_or(green, disease_mask)
    severity = cv2.countNonZero(disease_mask) / max(cv2.countNonZero(leaf), 1) * 100
    if severity < 20:
        level = "Hafif"
    elif severity < 50:
        level = "Orta"
    else:
        level = "Kritik"
    return severity, level

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yaprak Hastalığı Tahmini")
    parser.add_argument("image_path",           help="Tahmin yapılacak görüntü yolu")
    parser.add_argument("--model",    default="model.pth",        help="Model dosyası")
    parser.add_argument("--classes",  default="class_names.json", help="Sınıf isimleri")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Hata: Görüntü bulunamadı → {args.image_path}")
        exit(1)

    print("Model yükleniyor...")
    model, class_names = load_model(args.model, args.classes)

    print("Tahmin yapılıyor...")
    name, disease_tr, confidence = predict_image(args.image_path, model, class_names)
    severity, level = severity_analysis(args.image_path)

    print("\n========== SONUÇ ==========")
    print(f"Hastalık (EN): {name}")
    print(f"Hastalık (TR): {disease_tr}")
    print(f"Güven:         %{confidence*100:.1f}")
    print(f"Şiddet:        %{severity:.1f} ({level})")
    print("===========================")