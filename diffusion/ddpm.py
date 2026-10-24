import torch
from tqdm import tqdm


class DDPM:
    """DDPM 扩散过程封装：包含前向加噪、反向采样"""
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.T = T
        self.device = device

        # 1. 线性噪声调度
        self.beta = torch.linspace(beta_start, beta_end, T).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # 2. 预计算前向过程系数
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

        # 3. 预计算反向过程系数
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alpha)
        # 后验方差（DDPM原文公式）
        self.posterior_variance = (
            self.beta * (1.0 - torch.cat([torch.tensor([1.0]).to(device), self.alpha_bar[:-1]]))
            / (1.0 - self.alpha_bar)
        )

    def forward_diffusion(self, x0, t):
        """
        前向加噪：闭式公式直接计算t步带噪图像
        :param x0: 原始图像 [B, C, H, W]
        :param t: 时间步索引 [B]
        :return: 带噪图像x_t, 真实噪声noise
        """
        noise = torch.randn_like(x0)
        # 维度对齐到图像 [B, 1, 1, 1]
        sqrt_ab = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_1mab * noise
        return x_t, noise

    @torch.no_grad()
    def sample(self, model, n_samples, channels=1, img_size=28, save_process=True):
        """
        反向采样：从纯噪声逐步生成图像
        :param model: 噪声预测网络
        :param n_samples: 生成样本数量
        :param channels: 图像通道数
        :param img_size: 图像尺寸
        :param save_process: 是否保存中间步骤用于可视化
        :return: 生成图像x0, 中间过程列表
        """
        model.eval()
        # 初始化：标准高斯噪声
        x = torch.randn(n_samples, channels, img_size, img_size).to(self.device)
        process = []

        # 从 T-1 倒序迭代到 0
        for t in tqdm(reversed(range(self.T)), desc="反向采样", total=self.T):
            t_batch = torch.full((n_samples,), t, device=self.device).long()

            # 1. 网络预测噪声
            pred_noise = model(x, t_batch)

            # 2. 计算去噪后的均值
            x = self.sqrt_recip_alpha[t] * (
                x - (1 - self.alpha[t]) / self.sqrt_one_minus_alpha_bar[t] * pred_noise
            )

            # 3. 非最后一步添加随机噪声（增加生成多样性）
            if t > 0:
                z = torch.randn_like(x)
                sigma_t = torch.sqrt(self.posterior_variance[t])
                x = x + sigma_t * z

            # 保存中间过程
            if save_process and (t % 100 == 0 or t == self.T - 1 or t == 0):
                process.append(x.cpu())

        # 将[-1,1]范围映射回[0,1]
        x = (x + 1) / 2
        x = torch.clamp(x, 0.0, 1.0)
        return x, process
