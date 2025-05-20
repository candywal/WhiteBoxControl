# WhiteBoxControl

Remaining TODO
- [] Add links + emojis to make this nice




## File Structure of this Project:
WhiteBoxControl/
├── TrainProbe.py               # Main script for training probes
├── EvalProbe.py                # Main script for evaluating probes
├── Utils.py                    # Contains utils to download datasets, load models, etc.
|
|
├── probes/
│   ├── __init__.py
│   ├── base_probe.py           # Abstract base class for probes
│   ├── logistic_probe.py       # Implementation for Logistic Probe
│   ├── mean_diff_probe.py      # Implementation for Mean Difference Probe
│   └── attention_probe.py      # Implementation for Attention Probe
|
|
├── run_scripts/                      # (Optional but Recommended) Example configuration files
│   ├── example_bash_script.sh       # Example bash script to run the project  
|  
|
|
├── outputs/                      # (This directory is .gitignored)
│   ├── trained_probes/           # Default output for TrainProbe.py
│   │   └── <probe_output_name>/
│   │       ├── probe_weights.pt 
│   │       ├── config.json       # Hyperparams used for this probe, base model, probe_type
│   │       └── tokenizer_config/ 
|   | 
│   └── evaluation_results/       # Default output for EvalProbe.py
│       └── <eval_experiment_name>/
│           ├── metrics.json
│           ├── predictions.csv
│           └── plots/
|
├── requirements.txt              # Python dependencies
├── README.md                     # Project description, setup, usage instructions
└── .gitignore                    # Files and directories to ignore in Git