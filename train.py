import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import json, os, argparse

def train(data_dir, save_dir, epochs=10, batch_size=32, lr=0.001):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {DEVICE}")

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full = datasets.ImageFolder(data_dir)
    class_names = full.classes
    num_classes = len(class_names)
    print(f"Sınıf sayısı: {num_classes}, Toplam görüntü: {len(full)}")

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    train_size = int(0.8 * len(full))
    val_size = len(full) - train_size
    train_ds, val_ds = random_split(full, [train_size, val_size])
    train_ds.dataset.transform = train_tf
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        loss_sum, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * imgs.size(0)
            correct  += (out.argmax(1) == labels).sum().item()
            total    += labels.size(0)

        model.eval()
        vloss, vcorr, vtot = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                loss = criterion(out, labels)
                vloss += loss.item() * imgs.size(0)
                vcorr += (out.argmax(1) == labels).sum().item()
                vtot  += labels.size(0)

        train_acc = correct / total
        val_acc   = vcorr / vtot
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Train Loss: {loss_sum/total:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
            print(f"  → En iyi model kaydedildi (Val Acc: {val_acc:.4f})")

    print(f"\nEğitim tamamlandı. En iyi doğrulama doğruluğu: {best_val_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  default="data/PlantVillage", help="Veri seti klasörü")
    parser.add_argument("--save_dir",  default=".",                  help="Model kayıt klasörü")
    parser.add_argument("--epochs",    type=int, default=10)
    parser.add_argument("--batch_size",type=int, default=32)
    parser.add_argument("--lr",        type=float, default=0.001)
    args = parser.parse_args()
    train(args.data_dir, args.save_dir, args.epochs, args.batch_size, args.lr)