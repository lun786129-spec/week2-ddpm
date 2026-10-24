import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=2):
        super(VAE, self).__init__()
        
        # ========== 编码器 Encoder：对应变分后验 q(z|x) ==========
        # 输入：图片x  输出：后验分布的均值mu 和 对数方差log_var
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)      # 输出分布均值
        self.log_var_layer = nn.Linear(hidden_dim, latent_dim) # 输出分布对数方差（保证方差恒正）

        # ========== 解码器 Decoder：对应似然分布 p(x|z) ==========
        # 输入：隐变量z  输出：生成图片的像素值（归一化到0~1）
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid() # 把像素值压缩到0~1区间
        )

    def reparameterize(self, mu, log_var):
        # ========== 重参数化技巧 ==========
        # 解决：采样操作不可导，无法反向传播训练编码器
        # 公式：z = mu + sigma * epsilon，其中epsilon~N(0,1)
        std = torch.exp(0.5 * log_var)  # 标准差 = exp(对数方差/2)
        eps = torch.randn_like(std)     # 采样标准正态噪声（和模型参数无关，不需要梯度）
        return mu + eps * std

    def forward(self, x):
        # 1. 编码：得到后验分布的两个参数
        h = self.encoder(x)
        mu = self.mu_layer(h)
        log_var = self.log_var_layer(h)
        
        # 2. 重参数化采样出隐变量z
        z = self.reparameterize(mu, log_var)
        
        # 3. 解码：重构出图片
        recon_x = self.decoder(z)
        
        return recon_x, mu, log_var
