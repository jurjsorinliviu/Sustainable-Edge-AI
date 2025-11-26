"""
Chapter 4: Run ALL Sustainable Edge AI Experiments

This script runs all three-regime experiments for Chapter 4:
1. Burgers equation (three regimes)
2. Laplace equation (three regimes)
3. Memristor device (three regimes)

Each experiment compares:
- Continuous training (100% grid power, baseline)
- Passive solar training (50% duty, no adaptive reg)
- Active solar training (50% duty, with adaptive reg)

Usage:
    python run_all_chapter4_experiments.py
"""

import sys
import time
from pathlib import Path

# Add paths
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "chapter4" / "experiments"))

# Import all experiments
from three_regime_burgers_experiment import main as run_burgers
from three_regime_laplace_experiment import main as run_laplace
from three_regime_memristor_experiment import main as run_memristor


def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def main():
    """Run all Chapter 4 experiments"""
    
    print("="*80)
    print("CHAPTER 4: SUSTAINABLE EDGE AI - ALL EXPERIMENTS")
    print("Three-Regime Training Methodology")
    print("="*80)
    print("\nThis will run three complete experiments:")
    print("  1. Burgers Equation (PDE)")
    print("  2. Laplace Equation (PDE)")
    print("  3. Memristor Device (Compact Model)")
    print("\nEach experiment compares:")
    print("  • Continuous (100% grid power)")
    print("  • Passive (50% solar, no adaptive reg)")
    print("  • Active (50% solar, with adaptive reg)")
    print("\nEstimated total time: ~20-30 minutes")
    print("="*80)
    
    overall_start = time.time()
    all_results = {}
    
    # Experiment 1: Burgers Equation
    print_header("EXPERIMENT 1/3: BURGERS EQUATION")
    try:
        start = time.time()
        burgers_results = run_burgers()
        elapsed = time.time() - start
        all_results['burgers'] = {
            'status': 'SUCCESS',
            'time': elapsed,
            'results': burgers_results
        }
        print(f"\n✓ Burgers experiment completed in {elapsed:.1f}s")
    except Exception as e:
        print(f"\n✗ Burgers experiment FAILED: {e}")
        all_results['burgers'] = {'status': 'FAILED', 'error': str(e)}
    
    # Experiment 2: Laplace Equation
    print_header("EXPERIMENT 2/3: LAPLACE EQUATION")
    try:
        start = time.time()
        laplace_results = run_laplace()
        elapsed = time.time() - start
        all_results['laplace'] = {
            'status': 'SUCCESS',
            'time': elapsed,
            'results': laplace_results
        }
        print(f"\n✓ Laplace experiment completed in {elapsed:.1f}s")
    except Exception as e:
        print(f"\n✗ Laplace experiment FAILED: {e}")
        all_results['laplace'] = {'status': 'FAILED', 'error': str(e)}
    
    # Experiment 3: Memristor Device
    print_header("EXPERIMENT 3/3: MEMRISTOR DEVICE")
    try:
        start = time.time()
        memristor_results = run_memristor()
        elapsed = time.time() - start
        all_results['memristor'] = {
            'status': 'SUCCESS',
            'time': elapsed,
            'results': memristor_results
        }
        print(f"\n✓ Memristor experiment completed in {elapsed:.1f}s")
    except Exception as e:
        print(f"\n✗ Memristor experiment FAILED: {e}")
        all_results['memristor'] = {'status': 'FAILED', 'error': str(e)}
    
    # Summary
    overall_time = time.time() - overall_start
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*80)
    print(f"\nTotal execution time: {overall_time:.1f}s ({overall_time/60:.1f} minutes)")
    print("\nResults Summary:")
    print("-" * 80)
    
    for exp_name, exp_data in all_results.items():
        status_symbol = "✓" if exp_data['status'] == 'SUCCESS' else "✗"
        status_text = exp_data['status']
        time_text = f"{exp_data.get('time', 0):.1f}s" if 'time' in exp_data else "N/A"
        print(f"  {status_symbol} {exp_name.capitalize():15s}: {status_text:10s} (Time: {time_text})")
    
    print("-" * 80)
    
    # Success count
    success_count = sum(1 for v in all_results.values() if v['status'] == 'SUCCESS')
    total_count = len(all_results)
    print(f"\nSuccess rate: {success_count}/{total_count} experiments")
    
    if success_count == total_count:
        print("\n🎉 ALL EXPERIMENTS SUCCESSFUL!")
        print("\nResults are saved in chapter4/results/:")
        print("  • three_regime_burgers/")
        print("  • three_regime_laplace/")
        print("  • three_regime_memristor/")
        print("\nEach directory contains:")
        print("  - continuous_results.json")
        print("  - passive_results.json")
        print("  - active_results.json")
        print("  - regime_comparison.csv")
        print("  - regime_comparison.tex")
        print("  - regime_comparison.png")
        print("  - *_model.pth (trained models)")
    else:
        print("\n⚠️  Some experiments failed. Check error messages above.")
    
    return all_results


if __name__ == "__main__":
    results = main()