import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码：将离散时间步映射为高维连续向量"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResidualBlock(nn.Module):
    """残差卷积块，注入时间步嵌入信息"""
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

        # 时间嵌入投影
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))

        # 残差短路连接
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        # 注入时间信息
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)


class DownSample(nn.Module):
    """2倍下采样"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    """2倍上采样"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net 噪声预测网络：输入带噪图像+时间步，输出预测噪声"""
    def __init__(self, img_channels=1, base_dim=64, time_dim=256):
        super().__init__()
        self.time_embedding = SinusoidalPositionEmbeddings(time_dim)

        # 编码器（下采样路径）
        self.inc = nn.Conv2d(img_channels, base_dim, 3, padding=1)
        self.down_res1 = ResidualBlock(base_dim, base_dim * 2, time_dim)
        self.downsample1 = DownSample(base_dim * 2)
        self.down_res2 = ResidualBlock(base_dim * 2, base_dim * 4, time_dim)
        self.downsample2 = DownSample(base_dim * 4)

        # 瓶颈层
        self.bottleneck = ResidualBlock(base_dim * 4, base_dim * 4, time_dim)

        # 解码器（上采样路径 + 跳跃连接）
        self.upsample2 = UpSample(base_dim * 4)
        self.up_res2 = ResidualBlock(base_dim * 8, base_dim * 2, time_dim)
        self.upsample1 = UpSample(base_dim * 2)
        self.up_res1 = ResidualBlock(base_dim * 4, base_dim, time_dim)

        # 输出头
        self.out_conv = nn.Conv2d(base_dim, img_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)

        # 下采样
        x0 = self.inc(x)
        x1 = self.down_res1(x0, t_emb)
        x1_d = self.downsample1(x1)
        x2 = self.down_res2(x1_d, t_emb)
        x2_d = self.downsample2(x2)

        # 瓶颈
        h = self.bottleneck(x2_d, t_emb)

        # 上采样 + 跳跃连接
        h = self.upsample2(h)
        h = torch.cat([h, x2], dim=1)
        h = self.up_res2(h, t_emb)
        h = self.upsample1(h)
        h = torch.cat([h, x1], dim=1)
        h = self.up_res1(h, t_emb)

        return self.out_conv(h)
