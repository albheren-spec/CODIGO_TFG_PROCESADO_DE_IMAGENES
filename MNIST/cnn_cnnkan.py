import random, gc, time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from kan.KANLayer import KANLayer


SEED = 3
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


BATCH_SIZE, EPOCHS = 512, 20
transform = transforms.Compose([transforms.ToTensor()])

train_completo = datasets.MNIST("./data", train=True,
                                download=True, transform=transform)
test_ds        = datasets.MNIST("./data", train=False,
                                download=True, transform=transform)

val_size   = int(0.10 * len(train_completo))
train_size = len(train_completo) - val_size
train_ds, val_ds = random_split(
    train_completo, [train_size, val_size],
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


class KANLayerWrapper(nn.Module):
    def __init__(self, in_dim, out_dim, num=3, k=3):
        super().__init__()
        self.layer = KANLayer(in_dim, out_dim, num=num, k=k)

    def forward(self, x):
        out, _, _, _ = self.layer(x)
        return out

class CNNMLP(nn.Module):
    def __init__(self, cnn, flat_dim, mlp):
        super().__init__()
        self.cnn = cnn; self.flatten = nn.Flatten(); self.mlp = mlp

    def forward(self, x):
        return self.mlp(self.flatten(self.cnn(x)))

class CNNKAN(nn.Module):
    def __init__(self, cnn, flat_dim, kan):
        super().__init__()
        self.cnn = cnn; self.flatten = nn.Flatten(); self.kan = kan

    def forward(self, x):
        return self.kan(self.flatten(self.cnn(x)))


def crear_modelos():
    return {
        "CNNMLP_Debil": CNNMLP(
            nn.Sequential(nn.Conv2d(1, 1, 3), nn.ReLU()),
            flat_dim=1*26*26,
            mlp=nn.Sequential(nn.Linear(1*26*26, 10))
        ).to(DEVICE),

        "CNNMLP_Intermedio": CNNMLP(
            nn.Sequential(nn.Conv2d(1, 16, 3), nn.ReLU(), nn.MaxPool2d(2)),
            flat_dim=16*13*13,
            mlp=nn.Sequential(nn.Linear(16*13*13, 64), nn.ReLU(),
                               nn.Linear(64, 10))
        ).to(DEVICE),

        "CNNMLP_Fuerte": CNNMLP(
            nn.Sequential(
                nn.Conv2d(1, 32, 3), nn.ReLU(),
                nn.Conv2d(32, 64, 3), nn.ReLU(),
                nn.MaxPool2d(2), nn.Dropout(0.25),
                nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2)
            ),
            flat_dim=128*5*5,
            mlp=nn.Sequential(nn.Linear(128*5*5, 128), nn.ReLU(),
                               nn.Dropout(0.5), nn.Linear(128, 10))
        ).to(DEVICE),

        "CNNKAN_Debil": CNNKAN(
            nn.Sequential(nn.Conv2d(1, 1, 3), nn.ReLU()),
            flat_dim=1*26*26,
            kan=nn.Sequential(KANLayerWrapper(1*26*26, 10))
        ).to(DEVICE),

        "CNNKAN_Intermedio": CNNKAN(
            nn.Sequential(nn.Conv2d(1, 16, 3), nn.ReLU(), nn.MaxPool2d(2)),
            flat_dim=16*13*13,
            kan=nn.Sequential(KANLayerWrapper(16*13*13, 64),
                               KANLayerWrapper(64, 10))
        ).to(DEVICE),

        "CNNKAN_Fuerte": CNNKAN(
            nn.Sequential(
                nn.Conv2d(1, 32, 3), nn.ReLU(),
                nn.Conv2d(32, 64, 3), nn.ReLU(),
                nn.MaxPool2d(2), nn.Dropout(0.25),
                nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2)
            ),
            flat_dim=128*5*5,
            kan=nn.Sequential(KANLayerWrapper(128*5*5, 128),
                               KANLayerWrapper(128, 10))
        ).to(DEVICE),
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
    ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(
        cmap="Blues", ax=ax)
    plt.title(f"{nombre} -- Confusion Matrix")
    plt.tight_layout(); plt.show()

    # Ejemplos de errores
    errores   = np.where(preds != true)[0]
    x_test_np = test_ds.data.numpy()
    plt.figure(figsize=(10, 6))
    for i, idx in enumerate(errores[:9]):
        plt.subplot(3, 3, i+1)
        plt.imshow(x_test_np[idx], cmap='gray')
        plt.title(f"Real:{true[idx]} Pred:{preds[idx]}")
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
