from industrial_ad import (
    benchmark_runs,
    discover_run_dirs,
)


run_dirs = discover_run_dirs("runs/MLP-reconstruction", "runs/TCN-reconstruction", "runs/GRU-repeated-reconstruction", "runs/GRU-seq2seq-reconstruction", 'runs/Transformer-reconstruction')
# run_dirs += discover_run_dirs("runs/PCA-reconstruction")

benchmark_runs(
    run_dirs=run_dirs,
    checkpoint="best",
    benchmark_config={
        "enabled": True,
        "device": "cpu",
        "num_threads": 1,
        "warmup_runs": 50,
        "num_runs": 2000,
        "profile_memory": True,
    },
    skip_existing=False
)
