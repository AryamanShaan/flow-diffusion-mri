# train_glow.py
import os
from PIL import Image
import math
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils

from glow import Glow  # assumes glow.py is in the same folder

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_top_latent_shape(orig_shape, n_levels, squeeze_factor):
    C, H, W = orig_shape
    for level in range(n_levels):
        H = H // squeeze_factor
        W = W // squeeze_factor
        C = C * 4 if level == 0 else C * 2
        if level < n_levels - 1:
            C = C // 2  # split keeps half
    return (C, H, W)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=64)          # coupling subnet width
    parser.add_argument("--depth", type=int, default=3)           # steps per level
    parser.add_argument("--levels", type=int, default=2)          # number of levels
    parser.add_argument("--squeeze_factor", type=int, default=2)
    parser.add_argument("--decomp", type=str, default="NONE")     # your Conv2d1x1 setting
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="runs_glow_mnist")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    transform = transforms.Compose([
        transforms.ToTensor(),                               # [0,1]
        # (optional) tiny uniform dequantization:
        # transforms.Lambda(lambda x: (x + torch.rand_like(x)/256.).clamp(0., 1.)),
    ])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    # --- Model ---
    x_shape = (1, 28, 28)  # NCHW per-sample shape (MNIST grayscale)
    model = Glow(
        x_shape=x_shape,
        n_levels=args.levels,
        depth=args.depth,
        width=args.width,
        decomp=args.decomp,
        squeeze_factor=args.squeeze_factor,
        squeeze_type="chessboard",
        use_bias_1x1=False
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Training ---
    model.train()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        running_nll = 0.0
        for i, (x, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()

            nll, sd_z = model.loss(x)     # nll is mean over batch
            nll.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0) # TODO check
            optimizer.step()

            running_nll += nll.item()
            global_step += 1

            if (i + 1) % 100 == 0:
                avg = running_nll / 100
                # also print bits/dim (optional, for intuition)
                bpd = avg / (math.log(2.0) * x_shape[0] * x_shape[1] * x_shape[2])
                print(f"[Epoch {epoch} | Step {i+1:04d}] nll={avg:.3f} | bits/dim={bpd:.4f} | sd_z={sd_z.item():.4f}")
                running_nll = 0.0

        # save checkpoint each epoch
        ckpt_path = os.path.join(args.outdir, f"glow_mnist_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    # --- Sampling (5 images) ---
    model.eval()
    with torch.no_grad():
        samples = model.inverse(z=None, batch_size=5).detach().cpu()   # (5,1,28,28)
        samples = samples.clamp(0.0, 1.0)

        # Convert to uint8 [0,255] with proper rounding
        samples_u8 = (samples * 255.0).round().to(torch.uint8)        # (5,1,28,28)

        # Save each image as grayscale PNG ('L' mode)
        os.makedirs(args.outdir, exist_ok=True)
        for i in range(samples_u8.size(0)):
            img = samples_u8[i, 0].numpy()                            # (28,28), uint8
            Image.fromarray(img, mode='L').save(
                os.path.join(args.outdir, f"sample_uint8_{i+1}.png")
            )

if __name__ == "__main__":
    main()
