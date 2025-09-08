import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import math
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision.utils import save_image

# ------------------------ core noise function ---------------------------------
@torch.no_grad()
def add_heteroscedastic_gaussian(
    x,
    clamp=True
):
    """
    x:           (C,H,W) or (N,C,H,W) in [0,1]
    alpha_range: (low, high), sample alpha per pixel ~ Uniform
    delta_range: (low, high), sample delta per pixel ~ Uniform
    clamp:       clamp noisy image into [0,1]
    return_all:  if True, return (x, noise, y, std); else (y, noise, std)
    """
    # alpha_range=(0.1, 0.9)   # range for alpha (per pixel)
    # delta_range=(0.01, 0.09) # range for delta (per pixel)
    x = x.float()

    # sample per-pixel alpha and delta in the same shape as x
    # alpha = torch.empty_like(x).uniform_(alpha_range[0], alpha_range[1])
    # delta = torch.empty_like(x).uniform_(delta_range[0], delta_range[1])
    alpha = torch.full_like(x, 0.6)  
    delta = torch.full_like(x, 0.05)  


    # variance = alpha^2 * x + delta^2
    var = (alpha ** 2) * x + (delta ** 2)
    std = torch.sqrt(var)

    noise = torch.randn_like(x) * std
    y = x + noise

    if clamp:
        y = y.clamp(0.0, 1.0)

    return noise, y, x


# ------------------------ Creating Padded+Noisy Dataset -----------------------------
def precompute_mnist32_noise(save_path="mnist32_heteronoise_train.pt",
                             train=True,
                             batch_size=512,
                             clamp=True,
                             seed=123):
    """
    - Loads MNIST (N,1,28,28) in [0,1]
    - Pads to (N,1,32,32) with zeros (black)
    - Adds heteroscedastic Gaussian noise via your add_heteroscedastic_gaussian()
    - Saves a .pt with keys: 'x' (clean padded), 'noise', 'y' (noisy), 'clamp', 'seed', 'split'
    """
    torch.manual_seed(seed)

    ds = datasets.MNIST(root="data", train=train, download=True, transform=transforms.ToTensor())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    xs, noises, ys = [], [], []
    with torch.no_grad():  # your noise fn is also @no_grad; this just avoids accidental grads elsewhere
        for imgs, _ in dl:                       # imgs: (N,1,28,28) in [0,1]
            imgs = imgs.float()
            # pad (left, right, top, bottom) = (2,2,2,2) -> (N,1,32,32)
            imgs32 = F.pad(imgs, pad=(2, 2, 2, 2), mode='constant', value=0.0)
            

            noise, y, x = add_heteroscedastic_gaussian(imgs32, clamp=clamp)  # uses your exact function
            # print(x.min().item(), x.max().item())
            # print(f"min={x.min().item():.6f}  max={x.max().item():.6f}")

            xs.append(x.cpu()); noises.append(noise.cpu()); ys.append(y.cpu())

    out = {
        "x":     torch.cat(xs, 0),          # clean, padded to 32x32
        "noise": torch.cat(noises, 0),      # sampled noise
        "y":     torch.cat(ys, 0),          # noisy, padded
        "clamp": bool(clamp),
        "seed":  int(seed),
        "split": "train" if train else "test",
    }
    torch.save(out, save_path)
    print(f"Saved: {save_path}  shapes -> x:{out['x'].shape} noise:{out['noise'].shape} y:{out['y'].shape}")
    sample_root = os.path.join(os.path.dirname(save_path), "sample_heteroscedastic_noise")
    os.makedirs(sample_root, exist_ok=True)

    x_show = out["x"][:5]   # (5,1,32,32)
    y_show = out["y"][:5]   # (5,1,32,32)

    for i, (x_im, y_im) in enumerate(zip(x_show, y_show)):
        pair_dir = os.path.join(sample_root, f"{i:04d}")
        os.makedirs(pair_dir, exist_ok=True)
        # Save clean and noisy images separately
        save_image(x_im, os.path.join(pair_dir, "x.png"))  # clean (32x32)
        save_image(y_im, os.path.join(pair_dir, "y.png"))  # noisy (32x32)

    print(f"Wrote sample pairs to: {sample_root}")


def save_mnist32_uniform_noise_pairs(
    outdir: str,
    n: int = 5,
    train: bool = True,
    clamp: bool = True,
    seed: int | None = None,
    ):
    """
    - Loads the first `n` MNIST images (scaled to [0,1])
    - Pads each to (1, 32, 32) with zeros (black border)
    - Adds per-pixel Uniform[0,1] noise
    - Clamps noisy images to [0,1] if `clamp` is True
    - Saves per-sample folders under `outdir` with:
        clean.png          (clean 32x32 image)
        noisy.png          (clean + uniform noise, clamped if requested)
    """
    if seed is not None:
        torch.manual_seed(seed)

    # ToTensor -> [0,1]; Pad(2) -> 28x28 -> 32x32 with black padding
    transform32 = transforms.Compose([
        transforms.Pad(2, fill=0),
        transforms.ToTensor(),
    ])

    # First n images in deterministic order
    ds = datasets.MNIST(root="./data", train=train, download=True, transform=transform32)
    dl = DataLoader(ds, batch_size=n, shuffle=False, num_workers=0, pin_memory=False)
    imgs, _ = next(iter(dl))                 # (n,1,32,32), already in [0,1]
    imgs = imgs.clamp_(0.0, 1.0)

    # Uniform[0,1] per-pixel noise, same shape as imgs
    noise = torch.rand_like(imgs)
    noisy = imgs + noise
    if clamp:
        noisy.clamp_(0.0, 1.0)

    # Save pairs: outdir/0000/clean.png, outdir/0000/noisy.png, ...
    os.makedirs(outdir, exist_ok=True)
    for i in range(n):
        pair_dir = os.path.join(outdir, f"{i:04d}")
        os.makedirs(pair_dir, exist_ok=True)
        save_image(imgs[i],  os.path.join(pair_dir, "clean.png"))
        save_image(noisy[i], os.path.join(pair_dir, "noisy.png"))

    print(f"Saved {n} clean/noisy pairs to: {outdir}")
    
def save_mnist32_gaussian_noise_pairs(
    outdir: str,
    n: int = 5,
    train: bool = True,
    clamp: bool = True,
    sigma: float = 1.0,
    seed: int | None = None,
):
    """
    - Loads the first `n` MNIST images (scaled to [0,1])
    - Pads each to (1,32,32) with zero (black) border
    - Adds per-pixel Gaussian noise ~ N(0, sigma^2)
    - Clamps to [0,1] if `clamp` is True
    - Saves per-sample folders:
        <outdir>/0000/clean.png
        <outdir>/0000/noisy.png
        ...
    """
    if seed is not None:
        torch.manual_seed(seed)

    transform32 = transforms.Compose([
        transforms.Pad(2, fill=0),  # 28->32
        transforms.ToTensor(),      # -> [0,1], (1,32,32)
    ])

    ds = datasets.MNIST(root="./data", train=train, download=True, transform=transform32)
    dl = DataLoader(ds, batch_size=n, shuffle=False, num_workers=0, pin_memory=False)
    imgs, _ = next(iter(dl))              # (n,1,32,32), in [0,1]
    imgs = imgs.clamp_(0.0, 1.0)

    # Gaussian noise N(0, sigma^2 I)
    noise = torch.randn_like(imgs) * float(sigma)
    noisy = imgs + noise
    if clamp:
        noisy.clamp_(0.0, 1.0)

    os.makedirs(outdir, exist_ok=True)
    for i in range(n):
        pair_dir = os.path.join(outdir, f"{i:04d}")
        os.makedirs(pair_dir, exist_ok=True)
        save_image(imgs[i],  os.path.join(pair_dir, "clean.png"))
        save_image(noisy[i], os.path.join(pair_dir, "noisy.png"))

    print(f"Saved {n} clean/noisy Gaussian pairs to: {outdir}")

# For testing
def main():
    # precompute_mnist32_noise("mnist32_heteronoise_train.pt", train=True, clamp=True)
    # save_mnist32_uniform_noise_pairs(outdir="sample_uniform_noise", n=5, train=True, clamp=True, seed=123)
    save_mnist32_gaussian_noise_pairs(outdir="sample_gaussian_noise",n=5,train=True,clamp=True,sigma=1.0,seed=123)

if __name__ == "__main__":
    main()

# ********************************************************************************************************
# ********************************************************************************************************
# ********************************************************************************************************



# ------------------------ helpers to sample α, δ ------------------------------
# def sample_param_uniform(low, high, device):
#     """Sample a per-image scalar in [low, high], returned as shape (1,1,1)."""
#     val = torch.empty(1, device=device).uniform_(low, high)
#     return val.view(1, 1, 1)

# # ------------------------ dataset wrapper -------------------------------------
# class NoisyMNIST(Dataset):
#     """
#     Wraps torchvision MNIST and adds heteroscedastic Gaussian noise per sample.

#     You specify ranges for α and δ; each __getitem__ draws new scalars:
#         alpha ~ Uniform(alpha_min, alpha_max)
#         delta ~ Uniform(delta_min, delta_max)
#     """
#     def __init__(self, root="data", train=True,
#                  alpha_range=(0.1, 0.9),   # tweak
#                  delta_range=(0.01, 0.09)  # tweak
#                  ):
#         self.base = datasets.MNIST(root=root, train=train, download=True, transform=ToTensor())
#         self.alpha_min, self.alpha_max = alpha_range
#         self.delta_min, self.delta_max = delta_range

#     def __len__(self):
#         return len(self.base)

#     def __getitem__(self, idx):
#         clean, label = self.base[idx]      # (1,28,28) in [0,1]
#         device = clean.device

#         # sample per-image α, δ (scalars shaped (1,1,1) so they broadcast over (C,H,W))
#         alpha = sample_param_uniform(self.alpha_min, self.alpha_max, device)  # (1,1,1)
#         delta = sample_param_uniform(self.delta_min, self.delta_max, device)  # (1,1,1)

#         noisy, _, _ = add_heteroscedastic_gaussian(clean, alpha, delta, clamp=True)
#         # you might also want to *return* alpha, delta for logging/conditioning:
#         return noisy, clean, label

# # ------------------------ build loaders ---------------------------------------
# alpha_range = (0.1, 0.9)
# delta_range = (0.01, 0.09)

# train_ds = NoisyMNIST(train=True,  alpha_range=alpha_range, delta_range=delta_range)
# test_ds  = NoisyMNIST(train=False, alpha_range=alpha_range, delta_range=delta_range)

# train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
# test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

# # ------------------------ save 5 sample pairs ---------------------------------
# save_dir = "mnist_hetero_rand_params_samples"
# os.makedirs(save_dir, exist_ok=True)
# torch.manual_seed(0)
# idxs = torch.randperm(len(test_ds))[:5]

# for k, i in enumerate(idxs):
#     noisy, clean, _ = test_ds[i]
#     torchvision.utils.save_image(clean, f"{save_dir}/orig_{k}.png")
#     torchvision.utils.save_image(noisy, f"{save_dir}/noisy_{k}.png")

# print(f"Saved 5 (orig, noisy) pairs with random α,δ per image to: {save_dir}")

# ------------------------ (optional) sanity check -----------------------------
# Empirical vs theoretical variance on one image with resampling.
# with torch.no_grad():
#     clean0, _ = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())[0]
#     K = 256
#     Ys = []
#     for _ in range(K):
#         # draw fresh α, δ each time (per-image scalars)
#         a = sample_param_uniform(*alpha_range, clean0.device)
#         d = sample_param_uniform(*delta_range, clean0.device)
#         y, _, _ = add_heteroscedastic_gaussian(clean0, a, d, clamp=False)
#         Ys.append(y)
#     Y = torch.stack(Ys, 0)           # (K,1,28,28)
#     emp_var = Y.var(dim=0, unbiased=True)
#     # Since α,δ vary per draw, the theoretical variance for this mixture is E[α^2]*I + E[δ^2]
#     # (cross-terms drop because noise is zero-mean and independent of I)
#     a_low, a_high = alpha_range
#     d_low, d_high = delta_range
#     Ea2 = ((a_low**2 + a_low*a_high + a_high**2) / 3.0)  # E[U^2] for U~Uniform(a_low,a_high)
#     Ed2 = ((d_low**2 + d_low*d_high + d_high**2) / 3.0)
#     theo_var = Ea2 * clean0 + Ed2
#     print("mean abs var error (mixture):", (emp_var - theo_var).abs().mean().item())

