"""
绘制常用激活函数：Sigmoid 和 Softmax
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# ── Sigmoid ──────────────────────────────────────────────────────────────────
x = np.linspace(-10, 10, 500)
y_sigmoid = sigmoid(x)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(x, y_sigmoid, color="steelblue", linewidth=2)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="y = 0.5")
ax.axvline(0, color="gray", linestyle="--", linewidth=1, label="x = 0")
ax.set_title("Sigmoid Function", fontsize=14)
ax.set_xlabel("x")
ax.set_ylabel("σ(x)")
ax.set_ylim(-0.05, 1.05)
ax.legend()
ax.grid(True, alpha=0.3)

# 公式注释
ax.text(
    0.05, 0.92,
    r"$\sigma(x) = \dfrac{1}{1+e^{-x}}$",
    transform=ax.transAxes,
    fontsize=13,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
)

# ── Softmax ───────────────────────────────────────────────────────────────────
# 固定 3 个类别，令第 0 类的得分 x 在 [-6, 6] 变化，其余两类固定为 0
# 展示各类别概率随 x 变化的曲线
x_range = np.linspace(-6, 6, 500)
fixed = np.array([1.0, 2.0])          # 其余两类的固定得分
colors_cls = ["steelblue", "coral", "seagreen"]
labels_cls = ["class 0 (x varies)", "class 1 (score=1)", "class 2 (score=2)"]

probs_matrix = np.array([softmax(np.array([xi, *fixed])) for xi in x_range])

ax = axes[1]
for i, (color, label) in enumerate(zip(colors_cls, labels_cls)):
    ax.plot(x_range, probs_matrix[:, i], color=color, linewidth=2, label=label)

ax.set_title("Softmax Function (3 classes)", fontsize=14)
ax.set_xlabel("Score of class 0 (x)")
ax.set_ylabel("Probability")
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 公式注释
ax.text(
    0.05, 0.92,
    r"$\mathrm{softmax}(x_i) = \dfrac{e^{x_i}}{\sum_j e^{x_j}}$",
    transform=ax.transAxes,
    fontsize=13,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
)

plt.suptitle("Activation Functions", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("activations.png", dpi=150)
print("图像已保存至 activations.png")
plt.show()
