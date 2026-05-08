# Diffusion Models 学习系列

本目录是扩散模型的系统性学习仓库，各子目录共用 `model/` 中的基础组件，
仅在各自特有的采样/架构部分有所不同。

---

## 目录结构

```
dp/
├── model/                    # ★ 所有变体共用的基础组件
│   ├── model.py               # SimpleUNet（噪声预测骨干网络）
│   ├── noise_schedule.py      # β-schedule + 前向加噪 q_sample + 训练损失 p_losses
│   └── __init__.py
│
├── ddpm/                      # ✅ 已实现：DDPM 基础版
│   ├── sampler.py             # 祖先采样器（DDPM 特有）
│   ├── main.py                # 训练脚本
│   └── README.md
│
├── ddim/                      # 🔜 计划中：DDIM 确定性快速采样
│   ├── sampler.py             # 确定性采样器，支持跳步（DDIM 特有）
│   ├── main.py
│   └── README.md
│
├── stable_diffusion/          # 🔜 计划中：条件生成 / 潜空间扩散
│   └── ...
│
├── README.md                  # 本文件
└── requirements.txt
```

---

## model/ 模块说明

### `model/model.py` — SimpleUNet

所有扩散变体共用的噪声预测骨干网络：

```
输入: (B, C, H, W) 加噪图像 + (B,) 时间步 t
  → 时间步 Sinusoidal 嵌入 → MLP
  → 编码器（ResBlock + 下采样）
  → 瓶颈
  → 解码器（ResBlock + 上采样 + skip connection）
输出: (B, C, H, W) 预测噪声 ε_θ
```

### `model/noise_schedule.py` — NoiseSchedule

前向扩散过程，DDPM / DDIM 完全共用：

| 方法 | 公式 | 用途 |
|------|------|------|
| `q_sample(x0, t)` | $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon$ | 训练时加噪 |
| `p_losses(model, x0)` | $\|\varepsilon - \varepsilon_\theta(x_t,t)\|^2$ | 训练损失 |

支持两种 β 调度：
- `linear`：DDPM 原始线性调度
- `cosine`：Nichol & Dhariwal 2021 余弦调度（低时间步噪声更平滑）

---

## 各变体的核心区别

| 变体 | 反向采样 | 推理步数 | 条件输入 |
|------|---------|---------|---------|
| **DDPM** | 随机（每步加 σ_t·z） | 全部 T 步（慢） | 无 |
| **DDIM** | 确定性（无随机噪声） | 可跳步，50步即可 | 无 |
| **Stable Diffusion** | DDIM/DDPM 均可 | 跳步 | 文本/图像条件 |

---

## 快速开始

```bash
pip install -r requirements.txt

# 训练 DDPM
cd ddpm
python main.py --epochs 20

# 使用 cosine schedule（质量更好）
python main.py --epochs 20 --schedule cosine
```
