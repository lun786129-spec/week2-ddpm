import os
import torch
import matplotlib.pyplot as plt

from models import UNet
from diffusion import DDPM
from utils import get_mnist_dataloader


def main():
    # ===================== 配置参数 =====================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    single_img_idx = 0          # 单张全流程演示用的图片索引
    n_batch_samples = 16        # 批量重建对比的样本数量
    T = 1000                    # 扩散总步数，与训练一致
    weight_path = "./results/ddpm_mnist.pth"
    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)

    # 全流程展示的时间步节点
    show_steps = [0, 50, 100, 150, 200]

    # ===================== 加载模型与数据 =====================
    model = UNet(img_channels=1).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    ddpm = DDPM(T=T, device=device)

    _, dataset = get_mnist_dataloader()

    # ===================== 第一部分：单张图 加噪+去噪 全流程 =====================
    print("=== 正在生成单张全流程对比图 ===")
    x0_single, label_single = dataset[single_img_idx]
    x0_single = x0_single.unsqueeze(0).to(device)

    # 1. 前向加噪各步骤
    forward_imgs = []
    for t in show_steps:
        t_tensor = torch.tensor([t], device=device).long()
        x_t, _ = ddpm.forward_diffusion(x0_single, t_tensor)
        forward_imgs.append(x_t.cpu())

    # 2. 以最终加噪结果为起点，反向去噪重建
    x = forward_imgs[-1].to(device)
    reverse_imgs = []

    with torch.no_grad():
        for t in reversed(range(T)):
            t_batch = torch.tensor([t], device=device).long()
            pred_noise = model(x, t_batch)
            x = ddpm.sqrt_recip_alpha[t] * (
                x - (1 - ddpm.alpha[t]) / ddpm.sqrt_one_minus_alpha_bar[t] * pred_noise
            )
            if t > 0:
                z = torch.randn_like(x)
                sigma_t = torch.sqrt(ddpm.posterior_variance[t])
                x = x + sigma_t * z

            if t in show_steps:
                reverse_imgs.append(x.cpu())

    # 3. 绘制全流程对比图
    fig, axes = plt.subplots(2, len(show_steps), figsize=(16, 5))

    # 上排：前向加噪
    for i, t in enumerate(show_steps):
        img = forward_imgs[i].squeeze().numpy()
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].set_title(f"Noisy t={t}", fontsize=10)
        axes[0, i].axis("off")
    axes[0, 0].set_ylabel("Forward Process", fontsize=12)

    # 下排：反向去噪（顺序反转，与上排时间轴对齐）
    reverse_imgs = reverse_imgs[::-1]
    for i, t in enumerate(reversed(show_steps)):
        img = reverse_imgs[i].squeeze().numpy()
        axes[1, i].imshow(img, cmap="gray")
        axes[1, i].set_title(f"Denoised t={t}", fontsize=10)
        axes[1, i].axis("off")
    axes[1, 0].set_ylabel("Reverse Process", fontsize=12)

    plt.suptitle(f"Digit {label_single}: Full Noising & Reconstruction Pipeline", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "single_full_pipeline.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ===================== 第二部分：批量样本 原图vs重建图 对比 =====================
    print(f"=== 正在生成 {n_batch_samples} 张批量重建对比图 ===")
    # 1. 随机抽取n个样本索引
    rand_indices = torch.randint(0, len(dataset), (n_batch_samples,)).tolist()

    # 2. 取出原图并拼成batch
    x0_batch = []
    for idx in rand_indices:
        img, _ = dataset[idx]
        x0_batch.append(img)
    x0_batch = torch.stack(x0_batch).to(device)

    # 3. 一次性加噪到T步
    t_T = torch.full((n_batch_samples,), 500, device=device).long()
    x_T_batch, _ = ddpm.forward_diffusion(x0_batch, t_T)

    # 4. 批量反向去噪重建
    # 4. 批量反向去噪重建（从加噪终止步开始往回走）
    x = x_T_batch.clone()
    start_step = t_T[0].item()  # 以加噪的终止步作为反向起始点
    with torch.no_grad():
        for t in reversed(range(start_step + 1)):
            t_batch = torch.full((n_batch_samples,), t, device=device).long()
            pred_noise = model(x, t_batch)
            x = ddpm.sqrt_recip_alpha[t] * (
                x - (1 - ddpm.alpha[t]) / ddpm.sqrt_one_minus_alpha_bar[t] * pred_noise
            )
            if t > 0:
                z = torch.randn_like(x)
                sigma_t = torch.sqrt(ddpm.posterior_variance[t])
                x = x + sigma_t * z

    # 5. 归一化到[0,1]并移到CPU
    x0_show = (x0_batch.cpu() + 1) / 2
    x_rec_show = (x.cpu() + 1) / 2
    x_rec_show = torch.clamp(x_rec_show, 0.0, 1.0)

    # 6. 拼接：上排原图，下排重建图
    compare_batch = torch.cat([x0_show, x_rec_show], dim=0)
    from torchvision.utils import make_grid, save_image
    grid = make_grid(compare_batch, nrow=n_batch_samples, padding=2, normalize=False)
    save_image(grid, os.path.join(save_dir, "batch_original_vs_recon.png"), nrow=1)

    print("\n===== 全部生成完成 =====")
    print(f"单张全流程图: {save_dir}/single_full_pipeline.png")
    print(f"批量对比图: {save_dir}/batch_original_vs_recon.png")


if __name__ == "__main__":
    main()
