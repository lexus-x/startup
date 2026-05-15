#!/usr/bin/env python3
"""
Generate publication-quality plots from KSAM experiment results.
Usage: python scripts/plot_results.py --results results/ksam_results.json --output results/
"""

import argparse
import json
import os
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not installed. Skipping plots.")


def plot_success_comparison(results, output_dir):
    """Bar chart: Baseline vs SAC vs SAC+KSAM success rates."""
    methods = ['Baseline (BC)', 'SAC', 'SAC + KSAM']
    rates = [
        results['baseline']['success_rate'],
        results['sac']['success_rate'],
        results['sac_ksam']['success_rate'],
    ]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#95a5a6', '#3498db', '#e74c3c']
    bars = ax.bar(methods, rates, color=colors, edgecolor='black', linewidth=1.2)
    
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Success Rate', fontsize=13)
    ax.set_title('MetaWorld MT-10: Success Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, min(1.0, max(rates) * 1.3))
    ax.grid(axis='y', alpha=0.3)
    
    improvement = results['sac_ksam']['success_rate'] - results['sac']['success_rate']
    ax.annotate(f'KSAM: {improvement:+.1%}', xy=(2, rates[2]),
                xytext=(2, rates[2] + 0.08), fontsize=11, color='red',
                arrowprops=dict(arrowstyle='->', color='red'),
                ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: success_comparison.png")


def plot_singularity_analysis(results, output_dir):
    """Bar chart: Singularity failures comparison."""
    methods = ['SAC', 'SAC + KSAM']
    failures = [
        results['sac']['singularity_failures'],
        results['sac_ksam']['singularity_failures'],
    ]
    sing_eps = [
        results['sac']['singularity_episodes'],
        results['sac_ksam']['singularity_episodes'],
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Singularity episodes
    ax = axes[0]
    bars = ax.bar(methods, sing_eps, color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=1.2)
    for bar, val in zip(bars, sing_eps):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val}', ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Episodes Near Singularity')
    ax.set_title('Singularity Exposure', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Singularity-caused failures
    ax = axes[1]
    bars = ax.bar(methods, failures, color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=1.2)
    for bar, val in zip(bars, failures):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{val}', ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Failures from Singularities')
    ax.set_title('Singularity-Caused Failures', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    if failures[0] > 0:
        reduction = (1 - failures[1]/failures[0]) * 100
        axes[1].annotate(f'-{reduction:.0f}%', xy=(1, failures[1]),
                        xytext=(1, failures[1] + max(failures)*0.15),
                        fontsize=14, color='red', ha='center',
                        fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'singularity_analysis.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: singularity_analysis.png")


def plot_kappa_success_by_bin(results, output_dir):
    """Grouped bar chart: Success rate by kappa bin for each method."""
    bins = ['low(<10)', 'med(10-50)', 'high(50-100)', 'extreme(>100)']
    bin_labels = ['Low\n(κ<10)', 'Medium\n(10-50)', 'High\n(50-100)', 'Extreme\n(κ>100)']
    
    baseline_bins = results.get('success_by_kappa_bin', {}).get('baseline', {})
    sac_bins = results.get('success_by_kappa_bin', {}).get('sac', {})
    ksam_bins = results.get('success_by_kappa_bin', {}).get('sac_ksam', {})
    
    x = np.arange(len(bins))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baseline_vals = [baseline_bins.get(b, 0) for b in bins]
    sac_vals = [sac_bins.get(b, 0) for b in bins]
    ksam_vals = [ksam_bins.get(b, 0) for b in bins]
    
    ax.bar(x - width, baseline_vals, width, label='Baseline (BC)', color='#95a5a6', edgecolor='black')
    ax.bar(x, sac_vals, width, label='SAC', color='#3498db', edgecolor='black')
    ax.bar(x + width, ksam_vals, width, label='SAC + KSAM', color='#e74c3c', edgecolor='black')
    
    ax.set_xlabel('Singularity Exposure (Max κ in Episode)', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Success Rate by Singularity Exposure Level', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kappa_success_bins.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: kappa_success_bins.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results/ksam_results.json")
    parser.add_argument("--output", type=str, default="results/")
    args = parser.parse_args()
    
    if not HAS_MPL:
        print("Install matplotlib for plots: pip install matplotlib")
        return
    
    with open(args.results) as f:
        results = json.load(f)
    
    os.makedirs(args.output, exist_ok=True)
    
    print("Generating plots...")
    plot_success_comparison(results, args.output)
    plot_singularity_analysis(results, args.output)
    plot_kappa_success_by_bin(results, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
