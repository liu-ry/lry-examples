# DDPM（Denoising Diffusion Probabilistic Models）

本目录是 DDPM 的完整实现，基于 `dp/model/` 中的公共组件构建。

> 参考：Ho et al. *Denoising Diffusion Probabilistic Models* (NeurIPS 2020)

---

## DDPM 特有内容

| 文件 | 说明 |
|------|------|
| `sampler.py` | **祖先采样器**：每步去噪时注入随机噪声 $\sigma_t z$，需要走完全部 T 步 |
| `main.py`    | 训练入口，使用 `model/` 中的 UNet 和 NoiseSchedule |

---

## DDPM 反向采样公式

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta(x_t, t)\right) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)$$

- 每步都加随机噪声 $\sigma_t z = \sqrt{\beta_t}\, z$，因此是**随机采样**
- 完整推理需要走完全部 T 步（默认 1000 步）

## 与 DDIM 的对比（将在 `ddim/` 中实现）

| | DDPM | DDIM |
|-|------|------|
| 采样方式 | 随机（每步加噪） | 确定性 |
| 推理步数 | 需要全部 T 步 | 可跳步，50~100 步即可 |
| 训练 | 完全相同 | 完全相同 |
| 共用文件 | `model/model.py`, `model/noise_schedule.py` | 同左 |

---

## 运行

```bash
# 从 dp/ddpm/ 目录运行
cd dp/ddpm
python main.py

# 自定义参数
python main.py --epochs 50 --timesteps 1000 --schedule cosine

# cosine schedule（生成质量通常更好）
python main.py --schedule cosine
```

## 参数说明

```
--epochs        训练轮数           (default: 20)
--batch-size    batch 大小         (default: 128)
--lr            学习率              (default: 2e-4)
--timesteps     扩散总步数 T        (default: 1000)
--schedule      linear 或 cosine   (default: linear)
--log-interval  打印间隔            (default: 100)
--results-dir   图像保存目录        (default: results)
```

## 生成结果（保存在 `results/`）

- **`samples_epoch_N.png`**：4×4 网格，模型从纯噪声生成的手写数字
- **`denoising_epoch_N.png`**：去噪轨迹，从 $x_T$（纯噪声）到 $x_0$（清晰图像）约 10 帧
