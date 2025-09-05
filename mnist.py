# save_mnist_examples.py
import os
import random
import torch
from torchvision import datasets, transforms, utils as vutils

def main():
    outdir = "mnist_examples"
    os.makedirs(outdir, exist_ok=True)

    # --- Load MNIST test set ---
    transform = transforms.ToTensor()
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # --- Pick 5 random samples ---
    idxs = random.sample(range(len(test_ds)), 5)
    imgs = [test_ds[i][0] for i in idxs]  # each item is (image, label)

    # --- Save images ---
    for j, img in enumerate(imgs):
        path = os.path.join(outdir, f"mnist_original_{j+1}.png")
        vutils.save_image(img, path)
        print(f"Saved {path}")

    # also save as a grid for convenience
    grid_path = os.path.join(outdir, "mnist_original_grid.png")
    vutils.save_image(torch.stack(imgs, dim=0), grid_path, nrow=5, padding=2)
    print(f"Saved grid {grid_path}")

if __name__ == "__main__":
    main()
