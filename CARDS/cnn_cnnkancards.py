import random, gc, time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from kan.KANLayer import KANLayer
 

SEED = 3
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
 

DATASET_DIR = "/content/PlayCardsPalo" #Necesitas tener en el directorio el dataset, con las modificaciones expuestas en el trabajo
IMG_SIZE    = 128
BATCH_SIZE  = 64
EPOCHS      = 20
 
transform = Compose([Resize((IMG_SIZE, IMG_SIZE)), ToTensor()])
 
train_ds_completo = ImageFolder(f"{DATASET_DIR}/Train", transform=transform)
test_ds           = ImageFolder(f"{DATASET_DIR}/Test",  transform=transform)
 
val_size = int(0.10 * len(train_ds_completo))
train_ds, val_ds = random_split(
    train_ds_completo,
    [len(train_ds_completo) - val_size, val_size],
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
 

class KANLayerWrapper(nn.Module):
    def __init__(self, in_dim, out_dim, num=3, k=3):
        super().__init__()
        self.layer = KANLayer(in_dim, out_dim, num=num, k=k)
 
    def forward(self, x):
        out, _, _, _ = self.layer(x)
        return out
 
class RedHibrida(nn.Module):
    def __init__(self, cnn, flat_dim, clasificador):
        super().__init__()
        self.cnn          = cnn
        self.flatten      = nn.Flatten()
        self.clasificador = clasificador
 
    def forward(self, x):
        return self.clasificador(self.flatten(self.cnn(x)))
 
def cnn_intermedia():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2)
    )
 
def cnn_fuerte():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout2d(0.25),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(128, 16, 1), nn.ReLU()
    )
 
def crear_modelos():
    mlp_intermedio = nn.Sequential(
        nn.Linear(16*64*64, 64), nn.ReLU(),
        nn.Linear(64, NUM_CLASSES)
    )
    mlp_fuerte = nn.Sequential(
        nn.Linear(16*32*32, 128), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, NUM_CLASSES)
    )
    kan_intermedio = nn.Sequential(
        KANLayerWrapper(16*64*64, 64),
        KANLayerWrapper(64, NUM_CLASSES)
    )
    kan_fuerte = nn.Sequential(
        KANLayerWrapper(16*32*32, 128),
        KANLayerWrapper(128, NUM_CLASSES)
    )
    return {
        "CNNMLP_Intermedio": RedHibrida(cnn_intermedia(), 16*64*64,
                                        mlp_intermedio).to(DEVICE),
        "CNNMLP_Fuerte":     RedHibrida(cnn_fuerte(),     16*32*32,
                                        mlp_fuerte).to(DEVICE),
        "CNNKAN_Intermedio": RedHibrida(cnn_intermedia(), 16*64*64,
                                        kan_intermedio).to(DEVICE),
        "CNNKAN_Fuerte":     RedHibrida(cnn_fuerte(),     16*32*32,
                                        kan_fuerte).to(DEVICE),
    }
 

def entrenar(modelo, nombre):
    params  = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    opt     = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    history = {"train_acc": [], "val_acc": [], "loss": []}
    t0      = time.time()
 
    for epoch in range(EPOCHS):
        modelo.train()
        correct = total_loss = 0
 
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out  = modelo(x)
            loss = loss_fn(out, y)
            loss.backward(); opt.step()
            correct    += (out.argmax(1) == y).sum().item()
            total_loss += loss.item()
 
        modelo.eval()
        val_correct = 0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(DEVICE), yv.to(DEVICE)
                val_correct += (modelo(xv).argmax(1) == yv).sum().item()
 
        history["loss"].append(total_loss / len(train_loader))
        history["train_acc"].append(100 * correct / len(train_ds))
        history["val_acc"].append(100 * val_correct / len(val_ds))
 
    t_total = time.time() - t0
 
    # Evaluacion en test
    modelo.eval()
    preds, true = [], []
    with torch.no_grad():
        for xt, yt in test_loader:
            preds.extend(modelo(xt.to(DEVICE)).argmax(1).cpu().numpy())
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
    plt.title(f"{nombre} -- Confusion Matrix")
    plt.tight_layout(); plt.show()
 
    # Ejemplos de errores
    errores      = np.where(preds != true)[0]
    x_test_imgs  = [test_ds[i][0].permute(1, 2, 0).numpy()
                    for i in errores[:9]]
    plt.figure(figsize=(10, 6))
    for i, (img, idx) in enumerate(zip(x_test_imgs, errores[:9])):
        plt.subplot(3, 3, i+1)
        plt.imshow(np.clip(img, 0, 1))
        plt.title(f"Real:{class_names[true[idx]]} "
                  f"Pred:{class_names[preds[idx]]}")
        plt.axis('off')
    plt.suptitle(f"Errores -- {nombre}")
    plt.tight_layout(); plt.show()
 
    torch.save(modelo.state_dict(), f"{nombre}.pt")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
 
    return test_acc, t_total, params
 

modelos    = crear_modelos()
resultados = {}
for nombre, modelo in modelos.items():
    resultados[nombre] = entrenar(modelo, nombre)
