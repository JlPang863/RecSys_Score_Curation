"""Plot score transition matrix from denoise proxy labels report."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

report_path = "results/denoise_proxy_150k_tr030/skywork_fusion_mlp_150k_tr030_report.pt"
report = torch.load(report_path, weights_only=False)

T = report.diagnose['T']
if isinstance(T, torch.Tensor):
    T = T.numpy()

print("Transition matrix shape:", T.shape)
print(T)

fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(
    T,
    annot=True,
    fmt=".3f",
    cmap="YlGnBu",
    xticklabels=range(T.shape[1]),
    yticklabels=range(T.shape[0]),
    ax=ax,
    vmin=0,
    vmax=1,
    linewidths=0.5,
    linecolor='white',
)

ax.set_title("Score Transition Matrix\n(Proxy Labels, 150k pool, train_ratio=0.30)", fontsize=16)
ax.set_xlabel("Observed Score (Proxy Label)", fontsize=14)
ax.set_ylabel("True Score (Estimated)", fontsize=14)
ax.tick_params(labelsize=12)

plt.tight_layout()
out_path = "results/denoise_proxy_150k_tr030/transition_matrix.png"
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved to {out_path}")
plt.close()
