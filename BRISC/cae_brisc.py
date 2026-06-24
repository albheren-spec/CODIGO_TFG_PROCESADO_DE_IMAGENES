import random, time, gc, os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
 

SEED = 3
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
 

BASE_DIR    = "/content/brisc2025/classification_task"
SANOS_TRAIN = f"{BASE_DIR}/train/no_tumor"
SANOS_TEST  = f"{BASE_DIR}/test/no_tumor"
TUMOR_DIRS  = {
    "glioma":     f"{BASE_DIR}/test/glioma",
    "meningioma": f"{BASE_DIR}/test/meningioma",
    "pituitary":  f"{BASE_DIR}/test/pituitary"
}
 
IMG_SIZE   = 512
BATCH_SIZE = 16
EPOCHS     = 30
 

class ImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.transform = transform
 
    def __len__(self):
        return len(self.paths)
 
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        return img
 
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])
 
train_ds_completo = ImageDataset(SANOS_TRAIN, transform=transform)
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
 

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: 512x512 -> 256x256 -> 128x128 -> 64x64 -> 32x32
        self.encoder = nn.Sequential(
            nn.Conv2d(1,  4,  kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(4,  8,  kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8,  16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), nn.ReLU()
        )
        # Decoder: 32x32 -> 64x64 -> 128x128 -> 256x256 -> 512x512
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2,
                               padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 8,  kernel_size=3, stride=2,
                               padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(8,  4,  kernel_size=3, stride=2,
                               padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(4,  1,  kernel_size=3, stride=2,
                               padding=1, output_padding=1), nn.Sigmoid()
        )
 
    def forward(self, x):
        return self.decoder(self.encoder(x))
 
model   = ConvAutoencoder().to(DEVICE)
opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
 

history = {"train_loss": [], "val_loss": []}
 
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for x in train_loader:
        x = x.to(DEVICE)
        opt.zero_grad()
        loss = loss_fn(model(x), x)
        loss.backward(); opt.step()
        total_loss += loss.item()
 
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xv in val_loader:
            xv = xv.to(DEVICE)
            val_loss += loss_fn(model(xv), xv).item()
 
    history["train_loss"].append(total_loss / len(train_loader))
    history["val_loss"].append(val_loss / len(val_loader))
 
torch.save(model.state_dict(), "CAE_cerebros_sanos.pt")
 
plt.figure(figsize=(8, 4))
plt.plot(history["train_loss"], label="Train")
plt.plot(history["val_loss"],   label="Val")
plt.title("CAE - Loss"); plt.xlabel("Epoca"); plt.legend()
plt.tight_layout(); plt.show()
 

def detectar_tumor(img_path, umbral_pixel=None, titulo=""):
    img        = Image.open(img_path).convert('L')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
 
    model.eval()
    with torch.no_grad():
        reconstruccion = model(img_tensor)
 
    original     = img_tensor.squeeze().cpu().numpy()
    reconstruida = reconstruccion.squeeze().cpu().numpy()
    error_map    = np.abs(original - reconstruida)
 
    if umbral_pixel is None:
        umbral_pixel = error_map.mean() + 2 * error_map.std()
 
    mascara = (error_map > umbral_pixel).astype(np.uint8)
    alpha   = 0.6
    img_rgb = np.stack([original, original, original], axis=-1)
    overlay = img_rgb.copy()
    overlay[mascara == 1] = [1, 0, 0]
    img_rgb = (1 - alpha * mascara[..., np.newaxis]) * img_rgb + \
               alpha * mascara[..., np.newaxis] * overlay
 
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(original,     cmap='gray'); axes[0].set_title("Original")
    axes[1].imshow(reconstruida, cmap='gray'); axes[1].set_title("Reconstruccion")
    axes[2].imshow(error_map,    cmap='hot');  axes[2].set_title("Mapa de error")
    axes[3].imshow(img_rgb);                   axes[3].set_title("Anomalia detectada")
    for ax in axes: ax.axis('off')
    plt.suptitle(titulo); plt.tight_layout(); plt.show()
 

sanos_paths = [
    os.path.join(SANOS_TEST, f)
    for f in os.listdir(SANOS_TEST)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]
for img_path in sanos_paths[:3]:
    detectar_tumor(img_path, titulo="Sano")
 
for tipo, carpeta in TUMOR_DIRS.items():
    imagenes = [
        os.path.join(carpeta, f)
        for f in os.listdir(carpeta)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    for img_path in imagenes[:3]:
        detectar_tumor(img_path, titulo=f"Tumor: {tipo}")
 
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
