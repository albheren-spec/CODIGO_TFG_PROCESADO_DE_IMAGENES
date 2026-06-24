import random, time, gc, os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
 

SEED = 3
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
 

SEG_DIR     = "/content/brisc2025/segmentation_task"
TRAIN_IMGS  = f"{SEG_DIR}/train/images"
TRAIN_MASKS = f"{SEG_DIR}/train/masks"
TEST_IMGS   = f"{SEG_DIR}/test/images"
TEST_MASKS  = f"{SEG_DIR}/test/masks"
 
IMG_SIZE   = 512
BATCH_SIZE = 16
EPOCHS     = 35
 

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform_img=None,
                 transform_mask=None):
        self.img_paths  = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.mask_paths = sorted([
            os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.transform_img  = transform_img
        self.transform_mask = transform_mask
 
    def __len__(self):
        return len(self.img_paths)
 
    def __getitem__(self, idx):
        img  = Image.open(self.img_paths[idx]).convert('L')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        if self.transform_img:
            img  = self.transform_img(img)
        if self.transform_mask:
            mask = self.transform_mask(mask)
        mask = (mask > 0.5).float()
        return img, mask
 
transform_img = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
transform_mask = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
 
train_ds_completo = SegmentationDataset(
    TRAIN_IMGS, TRAIN_MASKS, transform_img, transform_mask)
test_ds = SegmentationDataset(
    TEST_IMGS,  TEST_MASKS,  transform_img, transform_mask)
 
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
 

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU()
        )
 
    def forward(self, x):
        return self.block(x)
 
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(1,   32)
        self.enc2 = ConvBlock(32,  64)
        self.enc3 = ConvBlock(64,  128)
        self.enc4 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(256, 512)
        self.up4  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(512, 256)
        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2  = nn.ConvTranspose2d(128,  64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1  = nn.ConvTranspose2d(64,   32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64,  32)
        self.out  = nn.Conv2d(32, 1, kernel_size=1)
 
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out(d1))
 
model = UNet().to(DEVICE)
 

def dice_loss(pred, target, eps=1e-6):
    intersection = (pred * target).sum(dim=(2, 3))
    union        = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - ((2 * intersection + eps) / (union + eps)).mean()
 
def dice_score(pred, target, threshold=0.5, eps=1e-6):
    pred_bin     = (pred > threshold).float()
    intersection = (pred_bin * target).sum(dim=(2, 3))
    union        = pred_bin.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return ((2 * intersection + eps) / (union + eps)).mean().item()
 

opt     = torch.optim.Adam(model.parameters(), lr=1e-4)
history = {"train_loss": [], "val_dice": []}
 
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for x, mask in train_loader:
        x, mask = x.to(DEVICE), mask.to(DEVICE)
        opt.zero_grad()
        loss = dice_loss(model(x), mask)
        loss.backward(); opt.step()
        total_loss += loss.item()
 
    model.eval()
    val_dice_acc = 0.0
    with torch.no_grad():
        for xv, maskv in val_loader:
            xv, maskv = xv.to(DEVICE), maskv.to(DEVICE)
            val_dice_acc += dice_score(model(xv), maskv)
 
    history["train_loss"].append(total_loss / len(train_loader))
    history["val_dice"].append(val_dice_acc / len(val_loader))
 
torch.save(model.state_dict(), "UNet_tumores.pt")
 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["train_loss"])
ax1.set_title("Dice Loss (Train)"); ax1.set_xlabel("Epoca")
ax2.plot(history["val_dice"])
ax2.set_title("Dice Score (Val)"); ax2.set_xlabel("Epoca")
plt.tight_layout(); plt.show()
 

model.eval()
test_dice_acc = 0.0
all_imgs, all_masks, all_preds, all_scores = [], [], [], []
 
with torch.no_grad():
    for x, mask in test_loader:
        x, mask = x.to(DEVICE), mask.to(DEVICE)
        out = model(x)
        test_dice_acc += dice_score(out, mask)
        for i in range(x.size(0)):
            all_imgs.append(x[i].cpu().numpy())
            all_masks.append(mask[i].cpu().numpy())
            all_preds.append(out[i].cpu().numpy())
            all_scores.append(dice_score(out[i:i+1], mask[i:i+1]))
 
test_dice_acc /= len(test_loader)
print(f"Test Dice Score: {test_dice_acc:.4f}")
 

indices = np.argsort(all_scores)
n       = len(indices)
idx_mostrar = [indices[0], indices[n//4], indices[n//2],
               indices[3*n//4], indices[-1]]
etiquetas   = ["Min", "Q1", "Q2", "Q3", "Max"]
 
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for col, (idx, etiqueta) in enumerate(zip(idx_mostrar, etiquetas)):
    img     = all_imgs[idx].squeeze()
    mask_np = all_masks[idx].squeeze()
    pred_np = (all_preds[idx].squeeze() > 0.5).astype(np.uint8)
    score   = all_scores[idx]
 
    axes[0, col].imshow(img, cmap='gray')
    axes[0, col].set_title(f"{etiqueta}\nDice: {score:.3f}")
    axes[0, col].axis('off')
 
    img_rgb = np.stack([img, img, img], axis=-1)
    overlay = img_rgb.copy()
    overlay[mask_np == 1]                    = [0, 0, 1]
    overlay[pred_np == 1]                    = [1, 0, 0]
    overlay[(mask_np == 1) & (pred_np == 1)] = [0, 1, 0]
    img_blend = np.clip((1 - 0.5) * img_rgb + 0.5 * overlay, 0, 1)
 
    axes[1, col].imshow(img_blend)
    axes[1, col].axis('off')
 
axes[0, 0].set_ylabel("Original",   fontsize=12)
axes[1, 0].set_ylabel("Prediccion", fontsize=12)
 
leyenda = [Patch(color='red',   label='Prediccion'),
           Patch(color='blue',  label='Ground truth'),
           Patch(color='green', label='Solapamiento')]
fig.legend(handles=leyenda, loc='lower center', ncol=3, fontsize=11)
plt.suptitle("U-Net: Ejemplos de prediccion por cuartil")
plt.tight_layout()
plt.subplots_adjust(bottom=0.08)
plt.show()
