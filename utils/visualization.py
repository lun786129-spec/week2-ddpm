import torch
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid


def plot_loss_curve(loss_history, save_path="./results/loss_curve.png"):
    """绘制并保存训练损失曲线"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, label="Training MSE Loss", color="#1f77b4")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DDPM Training Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_forward_process(ddpm, x0, save_path="./results/forward_process.png"):
    """可视化单张图像的前向加噪过程"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    show_steps = [0, 50, 100, 200, 400, 600, 800, 999]
    fig, axes = plt.subplots(1, len(show_steps), figsize=(16, 2.5))

    for idx, t in enumerate(show_steps):
        t_tensor = torch.tensor([t], device=x0.device).long()
        x_t, _ = ddpm.forward_diffusion(x0.unsqueeze(0), t_tensor)
        img = x_t.squeeze().cpu().numpy()
        axes[idx].imshow(img, cmap="gray")
        axes[idx].set_title(f"t = {t}", fontsize=10)
        axes[idx].axis("off")

    plt.suptitle("Forward Diffusion (加噪过程)", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_reverse_process(process, save_path="./results/reverse_process.png"):
    """可视化从噪声到图像的反向生成过程"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_steps = len(process)
    fig, axes = plt.subplots(1, num_steps, figsize=(16, 2.5))

    for i in range(num_steps):
        img = process[i][0].squeeze().numpy()
        axes[i].imshow(img, cmap="gray")
        axes[i].axis("off")

    plt.suptitle("Reverse Diffusion (去噪生成过程)", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_generated_images(images, save_path="./results/generated_samples.png", nrow=4):
    """保存生成的图像网格"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    grid = make_grid(images, nrow=nrow, padding=2, normalize=False)
    save_image(grid, save_path)
