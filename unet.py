# unet.py, Task 3, a small U-Net for binary nucleus segmentation.
#
# The network is deliberately small, base 16 channels and three downsampling
# steps, because the training set is only 80 images of 256x256. A full-size
# U-Net would have far more capacity than 80 images can constrain.
#
# Three losses are provided so the choice can be tested rather than assumed:
# Dice plus binary cross entropy, BCE alone, and Dice alone. See
# loss_ablation at the bottom of this file.
#
# This module needs torch, which is why it runs on Colab rather than with
# the rest of the pipeline.

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "outputs" / "models"


def build_unet(base=16, depth=3, in_ch=1):
    """Construct the small U-Net and return the module.

    With base=16 and depth=3 the channel widths are 16, 32, 64 on the way
    down and 128 at the bottleneck, roughly 0.5 M parameters. Each decoder
    stage upsamples by transposed convolution and concatenates the matching
    encoder feature map, which is what lets the network recover the nucleus
    boundaries lost to pooling.
    """
    import torch
    import torch.nn as nn

    def block(i, o):
        "Two 3x3 convolutions with batch norm, the standard U-Net unit."
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            widths = [base * 2 ** i for i in range(depth)]   # 16, 32, 64
            bottom = base * 2 ** depth                       # 128

            self.downs = nn.ModuleList()
            prev = in_ch
            for w in widths:
                self.downs.append(block(prev, w))
                prev = w
            self.bottleneck = block(prev, bottom)

            # decoder walks the widths in reverse, halving channels each step
            self.upconvs, self.ups = nn.ModuleList(), nn.ModuleList()
            prev = bottom
            for w in reversed(widths):
                self.upconvs.append(nn.ConvTranspose2d(prev, w, 2, stride=2))
                self.ups.append(block(w * 2, w))   # w upsampled + w skip
                prev = w

            self.head = nn.Conv2d(widths[0], 1, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            skips = []
            for down in self.downs:
                x = down(x)
                skips.append(x)
                x = self.pool(x)
            x = self.bottleneck(x)
            for upconv, up, skip in zip(self.upconvs, self.ups,
                                        reversed(skips)):
                x = upconv(x)
                x = torch.cat([x, skip], dim=1)
                x = up(x)
            return self.head(x)                 # logits, no sigmoid here

    return UNet()


def bce_loss(logits, target):
    """Binary cross entropy alone.

    Included as an ablation. About 92 per cent of pixels are background, so
    BCE can be driven low by a model that under-segments, which is exactly
    the behaviour the Dice term is meant to prevent.
    """
    import torch.nn.functional as F
    return F.binary_cross_entropy_with_logits(logits, target)


def soft_dice(logits, target, eps=1.0):
    "Soft Dice loss on the positive class (Milletari et al., 2016)."
    import torch
    probs = torch.sigmoid(logits)
    num = 2 * (probs * target).sum(dim=(1, 2, 3)) + eps
    den = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


def dice_bce_loss(logits, target, eps=1.0):
    "Dice plus BCE, the default, to cope with the heavy background imbalance."
    return bce_loss(logits, target) + soft_dice(logits, target, eps)


# selectable by name so the ablation can loop over them
LOSSES = {"dice_bce": dice_bce_loss, "bce": bce_loss, "dice": soft_dice}


def make_loaders(train_imgs, train_masks, val_imgs, val_masks, batch_size=8,
                 augment=True):
    """Wrap the numpy arrays in torch DataLoaders.

    Augmentation is flips and 90 degree rotations only. Nuclei have no
    canonical orientation, so these are label-preserving; brightness or
    elastic changes would risk altering the very intensity statistics the
    later feature extraction measures.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset

    class NucleiSet(Dataset):
        def __init__(self, images, masks, augment=False):
            self.images = images.astype(np.float32)
            self.masks = masks.astype(np.float32)
            self.augment = augment

        def __len__(self):
            return len(self.images)

        def __getitem__(self, i):
            img, mask = self.images[i], self.masks[i]
            if self.augment:
                k = np.random.randint(4)
                img, mask = np.rot90(img, k), np.rot90(mask, k)
                if np.random.rand() < 0.5:
                    img, mask = np.fliplr(img), np.fliplr(mask)
            return (torch.from_numpy(np.ascontiguousarray(img))[None],
                    torch.from_numpy(np.ascontiguousarray(mask))[None])

    train = DataLoader(NucleiSet(train_imgs, train_masks, augment),
                       batch_size=batch_size, shuffle=True)
    val = DataLoader(NucleiSet(val_imgs, val_masks, False),
                     batch_size=batch_size)
    return train, val


def train_unet(train_loader, val_loader, epochs=30, lr=1e-3, device=None,
               seed=0, verbose=True, loss_name="dice_bce", save=True):
    """Train the U-Net and return (model, history dataframe).

    loss_name selects from LOSSES, so the same routine serves the main run
    and the loss ablation. Keeps the weights from the epoch with the best
    validation Dice rather than the last epoch, so a late overfitting spike
    cannot be mistaken for the result.
    """
    import pandas as pd
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet().to(device)
    loss_fn = LOSSES[loss_name]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history, best_dice, best_state = [], -1.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss, dices = 0.0, []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += loss_fn(logits, y).item() * len(x)
                pred = (torch.sigmoid(logits) > 0.5).float()
                inter = (pred * y).sum(dim=(1, 2, 3))
                d = (2 * inter + 1e-7) / (pred.sum(dim=(1, 2, 3))
                                          + y.sum(dim=(1, 2, 3)) + 1e-7)
                dices.extend(d.cpu().numpy().tolist())
        val_loss /= len(val_loader.dataset)
        val_dice = float(np.mean(dices))
        history.append({"epoch": epoch, "loss_name": loss_name,
                        "train_loss": train_loss, "val_loss": val_loss,
                        "val_dice": val_dice})
        if val_dice > best_dice:
            best_dice = val_dice
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if verbose:
            print(f"epoch {epoch:3d}  train {train_loss:.4f}  "
                  f"val {val_loss:.4f}  val dice {val_dice:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    if save:
        MODELS.mkdir(parents=True, exist_ok=True)
        name = "unet.pt" if loss_name == "dice_bce" else f"unet_{loss_name}.pt"
        torch.save(model.state_dict(), MODELS / name)
    return model, pd.DataFrame(history)


def predict_masks(model, images, device=None, threshold=0.5, batch_size=8):
    "Predict boolean masks for a stack of images."
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = torch.from_numpy(
                images[i:i + batch_size].astype(np.float32))[:, None].to(device)
            probs = torch.sigmoid(model(batch))
            out.append((probs > threshold).cpu().numpy()[:, 0])
    return np.concatenate(out).astype(bool)


def load_unet(path=None, device=None):
    "Rebuild the network and load saved weights."
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet()
    model.load_state_dict(torch.load(path or (MODELS / "unet.pt"),
                                     map_location=device))
    return model.to(device).eval()


def loss_ablation(train_imgs, train_masks, val_imgs, val_masks, epochs=30,
                  seed=0):
    """Train the same architecture under each loss and compare on validation.

    Everything except the loss is held fixed (seed, architecture, optimiser,
    augmentation, epoch budget), so the difference in the resulting metrics
    is attributable to the loss alone.
    """
    import pandas as pd

    rows, histories = [], []
    for name in ["dice_bce", "bce", "dice"]:
        print(f"\n=== training with {name} ===", flush=True)
        train_loader, val_loader = make_loaders(train_imgs, train_masks,
                                                val_imgs, val_masks)
        model, history = train_unet(train_loader, val_loader, epochs=epochs,
                                    seed=seed, loss_name=name, verbose=False)
        preds = predict_masks(model, val_imgs)
        inter = np.logical_and(preds, val_masks).sum(axis=(1, 2))
        psum = preds.sum(axis=(1, 2))
        tsum = val_masks.sum(axis=(1, 2))
        dice = (2 * inter + 1e-7) / (psum + tsum + 1e-7)
        union = np.logical_or(preds, val_masks).sum(axis=(1, 2))
        iou = (inter + 1e-7) / (union + 1e-7)
        # a mask that under-segments still scores well on BCE, so record the
        # predicted nucleus area as a fraction of the true nucleus area
        coverage = psum.sum() / tsum.sum()
        rows.append({"loss": name, "val_dice": float(dice.mean()),
                     "val_iou": float(iou.mean()),
                     "predicted_area_ratio": float(coverage),
                     "best_epoch_dice": float(history.val_dice.max())})
        histories.append(history)
        print(f"{name}: dice {dice.mean():.4f} iou {iou.mean():.4f} "
              f"area ratio {coverage:.3f}", flush=True)
    return pd.DataFrame(rows), pd.concat(histories, ignore_index=True)
