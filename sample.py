import os
import argparse
import torch

from models import UNet
from diffusion import DDPM
from utils import visualize_reverse_process, save_generated_images


def parse_args():
    parser = argparse.ArgumentParser(description="DDPM MNIST 图像生成")
    parser.add_argument("--n_samples", type=int, default=16, help="生成样本数量")
    parser.add_argument("--T", type=int, default=1000, help="扩散总步数")
    parser.add_argument("--beta_start", type=float, default=1e-4, help="初始beta")
    parser.add_argument("--beta_end", type=float, default=0.02, help="最终beta")
    parser.add_argument("--img_size", type=int, default=28, help="图像尺寸")
    parser.add_argument("--weight_path", type=str, default="./results/ddpm_mnist.pth", help="模型权重路径")
    parser.add_argument("--save_dir", type=str, default="./results", help="结果保存目录")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"===== DDPM 图像生成启动 =====")
    print(f"生成数量: {args.n_samples} | 扩散步数: {args.T}")

    # 1. 初始化模型与扩散器
    model = UNet(img_channels=1).to(args.device)
    # 加载训练好的权重
    model.load_state_dict(torch.load(args.weight_path, map_location=args.device))
    print(f"已加载权重: {args.weight_path}")

    ddpm = DDPM(T=args.T, beta_start=args.beta_start, beta_end=args.beta_end, device=args.device)

    # 2. 反向采样生成
    generated_imgs, process = ddpm.sample(
        model,
        n_samples=args.n_samples,
        channels=1,
        img_size=args.img_size,
        save_process=True
    )

    # 3. 保存结果
    save_generated_images(generated_imgs, os.path.join(args.save_dir, "generated_samples.png"), nrow=4)
    visualize_reverse_process(process, os.path.join(args.save_dir, "reverse_process.png"))

    print(f"\n===== 生成完成 =====")
    print(f"生成样本: {args.save_dir}/generated_samples.png")
    print(f"生成过程可视化: {args.save_dir}/reverse_process.png")


if __name__ == "__main__":
    main()
