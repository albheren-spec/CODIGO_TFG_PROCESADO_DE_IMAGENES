import random, time, gc
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
 

SEED = 3
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
 

BASE_DIR  = "/content/brisc2025/classification_task"
BATCH_SIZE = 16
EPOCHS     = 20
 
transform_train = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
 
transform_test = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
 
train_ds_completo = ImageFolder(f"{BASE_DIR}/train", transform=transform_train)
test_ds           = ImageFolder(f"{BASE_DIR}/test",  transform=transform_test)
 
val_size   = int(0.10 * len(train_ds_completo))
train_size = len(train_ds_completo) - val_size
train_ds, val_ds = random_split(
    train_ds_completo, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)
 
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True,
                          persistent_workers=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True,
                          persistent_workers=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True,
                          persistent_workers=True)
 
class_names = train_ds_completo.classes
NUM_CLASSES = len(class_names)
 

def construir_densenet():
    model = models.densenet121(
        weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    return model
 
def construir_alexnet():
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(
        model.classifier[6].in_features, NUM_CLASSES)
    return model
 

def entrenar(model, nombre):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    history = {"train_acc": [], "val_acc": [], "loss": []}
    t0      = time.time()
 
    for epoch in range(EPOCHS):
        model.train()
        correct = total_loss = 0
 
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out  = model(x)
            loss = loss_fn(out, y)
            loss.backward(); opt.step()
            correct    += (out.argmax(1) == y).sum().item()
            total_loss += loss.item()
 
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(DEVICE), yv.to(DEVICE)
                val_correct += (model(xv).argmax(1) == yv).sum().item()
 
        history["loss"].append(total_loss / len(train_loader))
        history["train_acc"].append(100 * correct / len(train_ds))
        history["val_acc"].append(100 * val_correct / len(val_ds))
 
    t_total = time.time() - t0
 
    # Evaluacion en test
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for xt, yt in test_loader:
            preds.extend(model(xt.to(DEVICE)).argmax(1).cpu().numpy())
            true.extend(yt.numpy())
 
    preds, true = np.array(preds), np.array(true)
    test_acc = 100 * (preds == true).mean()
 
    # Curvas de entrenamiento
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_acc"], label="Train")
    ax1.plot(history["val_acc"],   label="Val")
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoca"); ax1.legend()
    ax2.plot(history["loss"])
    ax2.set_title("Loss"); ax2.set_xlabel("Epoca")
    plt.suptitle(nombre); plt.tight_layout(); plt.show()
 
    # Matriz de confusion
    cm = confusion_matrix(true, preds)
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        cmap="Blues", ax=ax)
    plt.title(f"{nombre}: Matriz de confusion")
    plt.tight_layout(); plt.show()
 
    torch.save(model.state_dict(), f"{nombre}.pt")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
 
    return test_acc, t_total, params
 

densenet = construir_densenet().to(DEVICE)
acc_d, t_d, params_d = entrenar(densenet, "DenseNet121")
 
alexnet = construir_alexnet().to(DEVICE)
acc_a, t_a, params_a = entrenar(alexnet, "AlexNet")
 
