import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


SEED = 3
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: 28x28x1 -> 14x14x16 -> 7x7x8
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # Decoder: 7x7x8 -> 14x14x8 -> 28x28x16 -> 28x28x1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
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

    for x, _ in train_loader:
        x = x.to(DEVICE)
        opt.zero_grad()
        loss = loss_fn(model(x), x)
        loss.backward(); opt.step()
        total_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xv, _ in val_loader:
            xv = xv.to(DEVICE)
            val_loss += loss_fn(model(xv), xv).item()

    history["train_loss"].append(total_loss / len(train_loader))
    history["val_loss"].append(val_loss / len(val_loader))


model.eval()
test_loss = 0.0
originals, reconstructions = [], []

with torch.no_grad():
    for x, _ in test_loader:
        x   = x.to(DEVICE)
        out = model(x)
        test_loss += loss_fn(out, x).item()
        if len(originals) < 10:
            originals.extend(x.cpu().numpy())
            reconstructions.extend(out.cpu().numpy())

test_loss /= len(test_loader)
print(f"Test MSE: {test_loss:.4f}")


plt.figure(figsize=(8, 4))
plt.plot(history["train_loss"], label="Train")
plt.plot(history["val_loss"],   label="Val")
plt.title("CAE -- Loss"); plt.xlabel("Epoca"); plt.legend()
plt.tight_layout(); plt.show()

fig, axes = plt.subplots(2, 10, figsize=(15, 3))
for i in range(10):
    axes[0, i].imshow(originals[i].squeeze(),       cmap='gray')
    axes[1, i].imshow(reconstructions[i].squeeze(), cmap='gray')
    axes[0, i].axis('off'); axes[1, i].axis('off')
axes[0, 0].set_ylabel("Original")
axes[1, 0].set_ylabel("Reconstruccion")
plt.suptitle("CAE MNIST -- Reconstrucciones")
plt.tight_layout(); plt.show()

torch.save(model.state_dict(), "CAE_MNIST.pt")
