"""
Generate Figure: Three-Regime Training Timeline
Shows power availability over 24h cycle for Continuous, Passive, and Active regimes
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300

# Create figure
fig, ax = plt.subplots(figsize=(10, 5))

# Time array (24 hours)
t = np.linspace(0, 24, 1000)

# Continuous regime: constant grid power
P_continuous = np.ones_like(t) * 250  # 250W constant

# Passive regime: 50% duty cycle (simplified solar model)
P_passive = np.where((t % 12) < 6, 300, 0)  # 6h on, 6h off, repeat

# Active regime: same power profile, but with adaptive regularization markers
P_active = P_passive.copy()

# Plot all three regimes
ax.plot(t, P_continuous, 'b-', linewidth=2.5, label='Continuous (Grid Power)', alpha=0.9)
ax.plot(t, P_passive, 'g-', linewidth=2.5, label='Passive (κ=0.0, Solar 50% duty cycle)', alpha=0.9)
ax.plot(t, P_active, 'r--', linewidth=2.5, label='Active (κ=2.0, Solar 50% duty cycle)', alpha=0.9)

# Mark checkpoint save points (at power transitions)
checkpoint_times = [6, 12, 18]  # Power interruption points
for tc in checkpoint_times:
    ax.axvline(x=tc, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax.plot(tc, 300, 'o', color='orange', markersize=10, markeredgewidth=2, 
            markerfacecolor='white', markeredgecolor='orange', zorder=10)

# Add text annotation for checkpoints
ax.text(6, 320, 'Checkpoint\nSaved', ha='center', va='bottom', fontsize=9, 
        bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

# Shade solar availability periods
solar_on_periods = [(0, 6), (12, 18)]
for start, end in solar_on_periods:
    ax.axvspan(start, end, alpha=0.1, color='yellow', label='Solar Available' if start == 0 else '')

# Add adaptive regularization indicator zones for Active regime
adaptive_zones = [(5, 6), (11, 12), (17, 18)]  # Last hour before power off
for start, end in adaptive_zones:
    ax.axvspan(start, end, alpha=0.15, color='red', 
               label='Adaptive ω↑' if start == 5 else '')

# Labels and formatting
ax.set_xlabel('Time (hours)', fontweight='bold')
ax.set_ylabel('Power Availability (W)', fontweight='bold')
ax.set_title('Three-Regime Training Timeline Over 24-Hour Cycle', 
             fontweight='bold', pad=15)

# Set axis limits
ax.set_xlim(0, 24)
ax.set_ylim(-20, 350)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Custom legend with better organization
ax.legend(loc='upper right', framealpha=0.95, edgecolor='black', ncol=2)

# Add annotations explaining key features
ax.annotate('GPU Power:\n250 W', 
            xy=(1, 250), xytext=(3, 180),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
            fontsize=9, color='blue', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.8))

ax.annotate('Solar Peak:\n300 W', 
            xy=(3, 300), xytext=(3, 250),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
            fontsize=9, color='green', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.8))

ax.annotate('Power Off:\nTraining Paused', 
            xy=(8, 0), xytext=(9, 50),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=9, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='mistyrose', alpha=0.8))

# Add text boxes for regime descriptions
textstr_passive = 'Passive: Standard\nregularization (ω)'
props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
ax.text(0.02, 0.70, textstr_passive, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)

textstr_active = 'Active: Adaptive\nregularization\nω×(1+κ×exp(...))'
props = dict(boxstyle='round', facecolor='lightcoral', alpha=0.5)
ax.text(0.02, 0.50, textstr_active, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)

# Add duty cycle annotation
ax.text(0.98, 0.95, '50% Duty Cycle\n(12h ON / 12h OFF)', 
        transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))

# Tight layout
plt.tight_layout()

# Save figure
output_path = 'chapter4/results/figure1_three_regime_timeline.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Figure saved to: {output_path}")

# Also save as PDF for publication
output_path_pdf = 'chapter4/results/figure1_three_regime_timeline.pdf'
plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
print(f"✓ Figure (PDF) saved to: {output_path_pdf}")

plt.show()

print("\nFigure Description:")
print("=" * 60)
print("This figure illustrates the three training regimes over a 24-hour")
print("cycle, showing how power availability differs between:")
print("  • Continuous: Constant 250W grid power (blue line)")
print("  • Passive: 50% duty cycle solar with standard regularization (green)")
print("  • Active: 50% duty cycle solar with adaptive regularization (red)")
print("")
print("Key features:")
print("  • Orange circles/lines: Checkpoint save points at power transitions")
print("  • Yellow shading: Solar power available periods")
print("  • Red shading: Adaptive regularization amplification zones")
print("  • Annotations: Explain power levels and training states")
print("=" * 60)