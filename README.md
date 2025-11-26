├── requirements.txt                # Python dependencies
├── run_all_experiments.py          # Master experiment runner
├── sustainable_edge_ai.py      # Main implementation
├── generate_figure1_timeline.py
├── generate_figure4_realistic_solar.py
├── run_all_chapter4_experiments.py
│
├── experiments/                # Individual problem experiments
│   ├── burgers_solar_experiment.py
│   ├── duty_cycle_sweep.py
│   ├── kappa_sweep_experiment.py
│   ├── realistic_solar_validation.py
│   ├── statistical_validation.py
│   ├── three_regime_advection_experiment.py
│   └── results/                    # Experimental outputs
│       ├── figure1_three_regime_timeline.png
│       ├── figure4_realistic_solar_comparison.png
│       ├── unified_results.csv
│       ├── unified_comparison_table.tex
│       ├── three_regime_burgers/
│       ├── three_regime_laplace/
│       ├── three_regime_memristor/
│       ├── statistical_validation/
│       ├── architecture_sensitivity/
│       ├── long_term_convergence/
│       └── realistic_solar_burgers/
│
├── PSI-HDL-implementation/          # Base Ψ-HDL framework
│   ├── Code/
│   │   ├── structure_extractor.py  # Hierarchical clustering
│   │   ├── verilog_generator.py    # HDL code generation
│   │   └── vteam_baseline.py       # Memristor baseline
│   └── Psi-NN-main/                # Original Ψ-NN framework
│       ├── Module/
│       │   ├── PsiNN_burgers.py
│       │   ├── PsiNN_laplace.py
│       │   ├── PsiNN_poisson.py
│       │   └── Training.py
│       └── Config/                 # Experiment configurations
│
├── results/                         # Legacy demonstration results
│   ├── burgers/
│   ├── memristor/
│   └── multi_model/
