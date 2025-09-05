import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import math
import torchvision
from torchvision import datasets, transforms

# ------------------------ core noise function ---------------------------------
@torch.no_grad()
def add_heteroscedastic_gaussian(
    x,
    clamp=True,
    return_all=False,
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

    if return_all:
        return x, noise, y, std
    return y, noise, std




# ------------------------ Creating Dataset -----------------------------
def precompute_mnist_noise(save_path="mnist_heteronoise_train.pt",
                           train=True,
                           batch_size=512,
                           clamp=True,
                           seed=123):
    torch.manual_seed(seed)

    ds = datasets.MNIST(root="data", train=train, download=True, transform=transforms.ToTensor())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    xs, noises, ys, stds = [], [], [], []
    for imgs, _ in dl:                  # imgs: (N,1,28,28) in [0,1]
        imgs = imgs.float()
        # a = torch.as_tensor(alpha, dtype=imgs.dtype, device=imgs.device)   # broadcastable
        # d = torch.as_tensor(delta, dtype=imgs.dtype, device=imgs.device)
        x, noise, y, std = add_heteroscedastic_gaussian(imgs, clamp=clamp, return_all=True)
        xs.append(x.cpu()); noises.append(noise.cpu()); ys.append(y.cpu()); stds.append(std.cpu())

    out = {
        "x": torch.cat(xs, 0),         # clean
        "noise": torch.cat(noises, 0), # sampled noise
        "y": torch.cat(ys, 0),         # noisy
        "std": torch.cat(stds, 0),     # per-pixel std
        # "alpha": float(alpha),
        # "delta": float(delta),
        "clamp": bool(clamp),
        "seed": int(seed),
        "split": "train" if train else "test",
    }
    torch.save(out, save_path)
    print(f"Saved: {save_path}  shapes -> x:{out['x'].shape} noise:{out['noise'].shape} y:{out['y'].shape} std:{out['std'].shape}")

# Example:
# precompute_mnist_noise("mnist_heteronoise_train.pt", train=True,  alpha=0.1, delta=0.01, clamp=False)
# precompute_mnist_noise("mnist_heteronoise_test.pt",  train=False, alpha=0.5, delta=0.01, clamp=False)




# ------------------------ Function for loading Dataset -----------------------------
class NoiseMaskDataset(Dataset):
    def __init__(self, pt_path, input_key="x", target_key="noise"):
        """
        input_key:  "x" (clean) or "y" (noisy) depending on your task
        target_key: "noise" or "std" (or even "y" for supervised denoising)
        """
        d = torch.load(pt_path, map_location="cpu")
        self.data = d
        self.x = d[input_key]    # (N,1,28,28)
        self.t = d[target_key]   # (N,1,28,28)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.t[idx]

# Example:
# train_ds = NoiseMaskDataset("mnist_heteronoise_train.pt", input_key="x", target_key="noise")
# train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

    

# For testing
def main():
    precompute_mnist_noise("mnist_heteronoise_train.pt", train=True, clamp=True)

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

