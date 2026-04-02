from industrial_ad import (
    benchmark_runs,
    discover_run_dirs,
)


run_dirs = discover_run_dirs("./runs")

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
)
