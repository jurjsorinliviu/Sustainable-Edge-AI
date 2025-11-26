"""
Main Runner Script for Chapter 4 Experiments
Executes all case studies and generates results for the manuscript
"""

import sys
import os
import time
import traceback

# Add src to path
sys.path.append('src')
sys.path.append('src/experiments')

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              SUSTAINABLE EDGE AI - CHAPTER 4 EXPERIMENTS                     ║
║                                                                              ║
║   Discovering Minimal Hardware Requirements Through Physics-Structure-      ║
║      Informed Learning Under Renewable Energy Constraints                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

def run_experiment(name, module_name, function_name):
    """Run a single experiment with error handling"""
    print(f"\n{'#'*80}")
    print(f"# {name}")
    print(f"{'#'*80}\n")
    
    start_time = time.time()
    
    try:
        # Import module
        module = __import__(module_name)
        
        # Get function
        if hasattr(module, function_name):
            func = getattr(module, function_name)
            
            # Run experiment
            results = func()
            
            elapsed = time.time() - start_time
            print(f"\n✓ {name} completed successfully in {elapsed:.1f} seconds")
            
            return results
        else:
            print(f"✗ Function {function_name} not found in {module_name}")
            return None
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {name} failed after {elapsed:.1f} seconds")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return None


def main():
    """Run all experiments"""
    
    all_results = {}
    experiment_times = {}
    
    total_start = time.time()
    
    # Experiment 1: Memristor Modeling (Grid vs Solar)
    print("\n" + "="*80)
    print("CASE STUDY 1: Memristor Compact Modeling")
    print("="*80)
    
    try:
        from experiments import memristor_experiment
        
        exp_start = time.time()
        
        # Run grid training
        print("\n[1/2] Running Grid-Powered Training...")
        results_grid = memristor_experiment.run_memristor_experiment(
            solar_constrained=False, 
            save_results=True
        )
        
        # Run solar training
        print("\n[2/2] Running Solar-Constrained Training...")
        results_solar = memristor_experiment.run_memristor_experiment(
            solar_constrained=True,
            save_results=True
        )
        
        all_results['memristor'] = {
            'grid': results_grid,
            'solar': results_solar
        }
        
        experiment_times['memristor'] = time.time() - exp_start
        print(f"\n✓ Memristor experiments completed in {experiment_times['memristor']:.1f}s")
        
    except Exception as e:
        print(f"\n✗ Memristor experiments failed: {str(e)}")
        traceback.print_exc()
    
    # Experiment 2: Burgers Equation PDE
    print("\n" + "="*80)
    print("CASE STUDY 2: Burgers Equation PDE Solving")
    print("="*80)
    
    try:
        from experiments import burgers_experiment
        
        exp_start = time.time()
        
        # Run grid training
        print("\n[1/2] Running Grid-Powered Training...")
        results_grid = burgers_experiment.run_burgers_experiment(
            solar_constrained=False,
            save_results=True
        )
        
        # Run solar training
        print("\n[2/2] Running Solar-Constrained Training...")
        results_solar = burgers_experiment.run_burgers_experiment(
            solar_constrained=True,
            save_results=True
        )
        
        all_results['burgers'] = {
            'grid': results_grid,
            'solar': results_solar
        }
        
        experiment_times['burgers'] = time.time() - exp_start
        print(f"\n✓ Burgers equation experiments completed in {experiment_times['burgers']:.1f}s")
        
    except Exception as e:
        print(f"\n✗ Burgers equation experiments failed: {str(e)}")
        traceback.print_exc()
    
    # Experiment 3: Driver Monitoring System
    print("\n" + "="*80)
    print("CASE STUDY 3: Driver Monitoring System Deployment")
    print("="*80)
    
    try:
        from experiments import driver_monitoring_experiment
        
        exp_start = time.time()
        
        results = driver_monitoring_experiment.run_driver_monitoring_case_study(
            save_results=True
        )
        
        all_results['driver_monitoring'] = results
        experiment_times['driver_monitoring'] = time.time() - exp_start
        
        print(f"\n✓ Driver monitoring case study completed in {experiment_times['driver_monitoring']:.1f}s")
        
    except Exception as e:
        print(f"\n✗ Driver monitoring case study failed: {str(e)}")
        traceback.print_exc()
    
    # Experiment 4: Multi-Model Generalization
    print("\n" + "="*80)
    print("CASE STUDY 4: Multi-Model Generalization Study")
    print("="*80)
    
    try:
        from experiments import multi_model_generalization
        
        exp_start = time.time()
        
        results = multi_model_generalization.run_multi_model_study(
            save_results=True
        )
        
        all_results['multi_model'] = results
        experiment_times['multi_model'] = time.time() - exp_start
        
        print(f"\n✓ Multi-model study completed in {experiment_times['multi_model']:.1f}s")
        
    except Exception as e:
        print(f"\n✗ Multi-model study failed: {str(e)}")
        traceback.print_exc()
    
    # Summary
    total_time = time.time() - total_start
    
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}\n")
    
    print("Completed Experiments:")
    for name, duration in experiment_times.items():
        print(f"  ✓ {name}: {duration:.1f}s")
    
    print(f"\nTotal execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    # Save consolidated results
    import numpy as np
    os.makedirs('results', exist_ok=True)
    np.save('results/all_experiments.npy', all_results)
    
    print(f"\n✓ All results saved to results/all_experiments.npy")
    
    # Print key findings
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}\n")
    
    if 'memristor' in all_results:
        mem = all_results['memristor']
        if 'grid' in mem and 'solar' in mem:
            cluster_reduction = (mem['grid']['n_clusters'] - mem['solar']['n_clusters']) / mem['grid']['n_clusters'] * 100
            cost_reduction = (mem['grid']['platform_cost'] - mem['solar']['platform_cost']) / mem['grid']['platform_cost'] * 100
            carbon_reduction = (mem['grid']['carbon_footprint']['total'] - mem['solar']['carbon_footprint']['total']) / mem['grid']['carbon_footprint']['total'] * 100
            
            print("Memristor Modeling:")
            print(f"  • Cluster reduction: {cluster_reduction:.1f}%")
            print(f"  • Cost reduction: {cost_reduction:.1f}%")
            print(f"  • Carbon reduction: {carbon_reduction:.1f}%")
            print(f"  • Grid platform: {mem['grid']['best_platform']}")
            print(f"  • Solar platform: {mem['solar']['best_platform']}")
    
    if 'driver_monitoring' in all_results:
        dm = all_results['driver_monitoring']
        print(f"\nDriver Monitoring System:")
        print(f"  • Recommended: {dm['recommended_platform']}")
        print(f"  • Baseline: {dm['baseline_platform']}")
        print(f"  • Savings per unit: ${dm['cost_savings_per_unit']:.2f}")
        print(f"  • Savings at 10k units: ${dm['cost_savings_10k_units']:,.2f}")
    
    if 'multi_model' in all_results:
        mm = all_results['multi_model']
        print(f"\nMulti-Model Generalization:")
        print(f"  • Models tested: {mm['n_models']}")
        print(f"  • Platform prediction accuracy: {mm['tier_match_rate']:.1f}%")
        print(f"  • Average cost savings: {mm['cost_savings_pct']:.1f}%")
        print(f"  • Average cluster reduction: {mm['avg_cluster_reduction']:.1f}%")
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}\n")
    
    return all_results


if __name__ == "__main__":
    results = main()