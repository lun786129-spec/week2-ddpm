import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from model import VAE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ========== 配置 ==========
LATENT_DIM = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = 'output/generations'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载训练好的最终模型
model = VAE(latent_dim=LATENT_DIM).to(DEVICE)
model.load_state_dict(torch.load('output/vae_final.pth', map_location=DEVICE))
model.eval()

# ========== 1. 随机生成图像：从先验p(z)=N(0,1)采样 ==========
def generate_random(n_samples=64):
    print('生成随机样本...')
    # 从标准正态先验里随机采样z
    z = torch.randn(n_samples, LATENT_DIM).to(DEVICE)
    
    with torch.no_grad():
        gen_imgs = model.decoder(z)
    
    gen_imgs = gen_imgs.view(-1, 1, 28, 28)
    save_image(gen_imgs, f'{OUTPUT_DIR}/random_generation.png', nrow=8, normalize=True)

# ========== 2. 隐空间插值：验证隐空间连续性 ==========
def latent_interpolation(steps=10):
    print('生成隐空间插值序列...')
    # 定义隐空间的两个端点
    z_start = torch.tensor([-2, -2], dtype=torch.float32).to(DEVICE)
    z_end = torch.tensor([2, 2], dtype=torch.float32).to(DEVICE)
    
    # 线性插值生成中间的z
    z_list = []
    for alpha in np.linspace(0, 1, steps):
        z = z_start * (1 - alpha) + z_end * alpha
        z_list.append(z)
    z_tensor = torch.stack(z_list, dim=0)
    
    with torch.no_grad():
        gen_imgs = model.decoder(z_tensor)
    
    gen_imgs = gen_imgs.view(-1, 1, 28, 28)
    save_image(gen_imgs, f'{OUTPUT_DIR}/interpolation.png', nrow=steps, normalize=True)

# ========== 3. 隐空间散点图：查看后验分布 ==========
def visualize_latent_space():
    print('绘制隐空间分布图...')
    # 加载测试集
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)
    
    all_z = []
    all_labels = []
    
    with torch.no_grad():
        for x, labels in test_loader:
            x = x.view(-1, 784).to(DEVICE)
            h = model.encoder(x)
            mu = model.mu_layer(h)  # 取后验的均值作为z的代表
            all_z.append(mu.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_z = np.concatenate(all_z, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 画散点图，不同数字用不同颜色
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(all_z[:, 0], all_z[:, 1], c=all_labels, cmap='tab10', s=1, alpha=0.7)
    plt.colorbar(scatter, ticks=range(10), label='数字类别')
    plt.title('VAE 隐空间后验分布')
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.grid(alpha=0.3)
    plt.savefig(f'{OUTPUT_DIR}/latent_space.png', dpi=300, bbox_inches='tight')
    plt.close()

# ========== 主程序 ==========
if __name__ == '__main__':
    generate_random()
    latent_interpolation()
    visualize_latent_space()
    print('\n所有可视化完成！结果都在 output/generations/ 文件夹里')
