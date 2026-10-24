import os
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models import UNet
from diffusion import DDPM
from utils import get_mnist_dataloader, plot_loss_curve, visualize_forward_process


class EMA:
    """
    指数移动平均 (Exponential Moving Average)
    扩散模型标准配置：平滑模型权重，显著提升生成图像的清晰度与自然度
    """
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}  # 影子权重（平滑后的权重）
        self.backup = {}  # 原始权重备份
        self.register()

    def register(self):
        """初始化影子权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """每步更新影子权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1 - self.decay) * param.data
                ).clone()

    def apply_shadow(self):
        """切换为平滑权重，用于保存/采样"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """恢复原始权重，继续训练"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


def parse_args():
    parser = argparse.ArgumentParser(description="DDPM MNIST 训练-优化版")
    # 训练超参
    parser.add_argument("--epochs", type=int, default=40, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
    parser.add_argument("--lr", type=float, default=2e-4, help="初始学习率")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA平滑系数")
    # 扩散超参
    parser.add_argument("--T", type=int, default=1000, help="扩散总步数")
    parser.add_argument("--beta_start", type=float, default=1e-4, help="初始beta")
    parser.add_argument("--beta_end", type=float, default=0.02, help="最终beta")
    parser.add_argument("--img_size", type=int, default=28, help="图像尺寸")
    # 路径
    parser.add_argument("--save_dir", type=str, default="./results", help="结果保存目录")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"===== DDPM MNIST 训练启动（优化版）=====")
    print(f"运行设备: {args.device}")
    print(f"训练轮数: {args.epochs} | 批次大小: {args.batch_size} | 初始学习率: {args.lr}")
    print(f"梯度裁剪: {args.grad_clip} | EMA系数: {args.ema_decay} | 扩散步数: {args.T}\n")

    # 1. 加载数据
    dataloader, dataset = get_mnist_dataloader(args.batch_size, args.img_size)

    # 2. 初始化模型、优化器、EMA、学习率调度器
    model = UNet(img_channels=1).to(args.device)
    ddpm = DDPM(T=args.T, beta_start=args.beta_start, beta_end=args.beta_end, device=args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    ema = EMA(model, decay=args.ema_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 3. 训练循环
    loss_history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for x0, _ in pbar:
            x0 = x0.to(args.device)
            cur_batch = x0.shape[0]

            # 随机采样时间步
            t = torch.randint(0, args.T, (cur_batch,), device=args.device).long()
            # 前向加噪
            x_t, noise = ddpm.forward_diffusion(x0, t)
            # 预测噪声 + 计算MSE损失
            pred_noise = model(x_t, t)
            loss = nn.functional.mse_loss(pred_noise, noise)

            # 反向传播 + 梯度裁剪（彻底解决loss爆炸）
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            # 更新EMA平滑权重
            ema.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")

        # 学习率退火
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1:2d}  平均损失: {avg_loss:.6f}\n")

    # 4. 保存权重
    # 普通训练权重
    torch.save(model.state_dict(), os.path.join(args.save_dir, "ddpm_mnist.pth"))
    # EMA平滑权重（采样推荐用这个，生成质量更好）
    ema.apply_shadow()
    torch.save(model.state_dict(), os.path.join(args.save_dir, "ddpm_mnist_ema.pth"))
    ema.restore()

    # 5. 保存损失曲线与可视化
    plot_loss_curve(loss_history, os.path.join(args.save_dir, "loss_curve.png"))
    sample_x0, _ = dataset[0]
    sample_x0 = sample_x0.to(args.device)
    visualize_forward_process(ddpm, sample_x0, os.path.join(args.save_dir, "forward_process.png"))

    print(f"===== 训练完成 =====")
    print(f"普通权重: {args.save_dir}/ddpm_mnist.pth")
    print(f"EMA平滑权重: {args.save_dir}/ddpm_mnist_ema.pth（推荐采样使用）")
    print(f"损失曲线: {args.save_dir}/loss_curve.png")
    print(f"加噪可视化: {args.save_dir}/forward_process.png")


if __name__ == "__main__":
    main()
