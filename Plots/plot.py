import matplotlib.pyplot as plt
import numpy as np

# Labels
labels = ['golf ball', 'overall avg']

# Finetuned (unchanged)
finetuned = np.array([0.6560,0.6048,0.7604,0.7112,0.7252,0.6872,0.7444,0.6916,0.7200,0.6676])

# 
proposed = np.array([
    0.5524, 0.5784, 0.6388, 0.6788, 0.6596,
    0.6596, 0.6376, 0.5564, 0.5400, 0.4860
])

#
esd_u = np.array([
    0.6540, 0.5356, 0.7256, 0.6092, 0.6564,
    0.6436, 0.6856, 0.6320, 0.6812, 0.6440
])

# Index for golf ball
gb_idx = 9

# Build new arrays: [golf_ball, average]
finetuned_new = [finetuned[gb_idx], finetuned.mean()]
esd_u_new     = [esd_u[gb_idx], esd_u.mean()]
proposed_new  = [proposed[gb_idx], proposed.mean()]

x = np.arange(len(labels))
w = 0.25

fig, ax = plt.subplots(figsize=(6, 4))

# Plot bars
bars1 = ax.bar(x - w, finetuned_new, w, label='After fine-tuning', color='#378ADD')
bars2 = ax.bar(x,     esd_u_new,     w, label='After ESD-U',       color='#E24B4A')
bars3 = ax.bar(x + w, proposed_new,  w, label='After proposed method', color='#1D9E75')

# Function to add labels
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                height + 0.005,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

# Add value labels
add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# Formatting
ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.2, label='Random chance (0.5)')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0.45, 0.82)
ax.set_ylabel('CLiD MIA AUC')
ax.set_title('MIA AUC (Golf Ball vs Overall Average)')
ax.legend(fontsize=9)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('mia_golfball_summary.png', dpi=300)
plt.show()