import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model import VAE

# ========== 超参数配置 ==========
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 1e-3
LATENT_DIM = 2  # 设为2维方便后续可视化隐空间
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== 自动创建输出目录 ==========
os.makedirs('output/reconstructions', exist_ok=True)
os.makedirs('output/checkpoints', exist_ok=True)

# ========== 加载MNIST数据集（自动下载） ==========
transform = transforms.Compose([
    transforms.ToTensor(),  # 转张量 + 自动归一化到0~1
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ========== 初始化模型和优化器 ==========
model = VAE(latent_dim=LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ========== VAE损失函数：重构损失 + KL散度损失 ==========
def vae_loss(recon_x, x, mu, log_var):
    # 1. 重构损失：衡量生成图和原图的像素差异（对应负对数似然）
    recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # 2. KL散度损失：约束后验q(z|x)接近先验p(z)=N(0,1)
    # 两个对角高斯分布KL散度的解析解
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + kl_loss, recon_loss, kl_loss

# ========== 单轮训练 ==========
def train_one_epoch():
    model.train()
    total_loss = 0
    
    for x, _ in train_loader:
        x = x.view(-1, 784).to(DEVICE)  # 把28x28图片展平成784维向量
        
        optimizer.zero_grad()
        recon_x, mu, log_var = model(x)
        loss, _, _ = vae_loss(recon_x, x, mu, log_var)
        
        loss.backward()  # 反向传播
        optimizer.step() # 更新参数
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader.dataset)

# ========== 测试 + 保存重构对比图 ==========
def test_and_save(epoch):
    model.eval()
    
    with torch.no_grad():
        # 取第一批测试图片做重构可视化
        x, _ = next(iter(test_loader))
        x = x.view(-1, 784).to(DEVICE)
        recon_x, _, _ = model(x)
        
        # 拼接原图和重构图画成对比网格
        n = 8  # 取前8张对比
        x_imgs = x.view(-1, 1, 28, 28)[:n]
        recon_imgs = recon_x.view(-1, 1, 28, 28)[:n]
        compare = torch.cat([x_imgs, recon_imgs], dim=0)
        save_image(compare, f'output/reconstructions/epoch_{epoch:02d}.png', nrow=n, normalize=True)

# ========== 主训练流程 ==========
if __name__ == '__main__':
    print(f'使用设备: {DEVICE}')
    print('开始训练VAE...\n')
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch()
        test_and_save(epoch)
        
        print(f'Epoch {epoch:02d} | 平均损失: {train_loss:.4f}')
        
        # 保存模型权重
        torch.save(model.state_dict(), f'output/checkpoints/vae_epoch_{epoch:02d}.pth')
    
    # 保存最终模型
    torch.save(model.state_dict(), 'output/vae_final.pth')
    print('\n训练完成！')
    print('最终模型：output/vae_final.pth')
    print('重构对比图：output/reconstructions/ 文件夹')
