from utils.pareto import plot_pareto_per_compressor, get_valid_studies, get_pareto_front_multiple_studies, get_summary_studies
from utils.optimizer import OptimizationRunIn
import os
import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
    # Load PostgreSQL configuration
    POSTGRES_CONFIG = OptimizationRunIn.load_postgres_config()

    output_dir = os.path.join("Results", "pareto_figs")
    os.makedirs(output_dir, exist_ok=True)

    # Get database URL (SQLite or PostgreSQL)
    storage_name = (f"postgresql://{POSTGRES_CONFIG['user']}:{POSTGRES_CONFIG['password']}"
                    f"@{POSTGRES_CONFIG['host']}:{POSTGRES_CONFIG['port']}"
                    f"/{POSTGRES_CONFIG['database']}")
    
    compressor_models = ["a", "b", "all"]

    studies = get_valid_studies(storage_name)
    if len(studies) == 0:
        raise ValueError("No valid studies found in the database.")
    summaries = get_pareto_front_multiple_studies(studies, storage_name)
    
    summary = ""
    result = []
    for compressor_model in compressor_models:
        r,s = get_summary_studies(summaries=summaries, compressor_model=compressor_model)
        summary += s
        result.extend(r)
        fig, ax = plot_pareto_per_compressor(summaries=summaries, compressor_model=compressor_model)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"pareto_plot_{compressor_model}.png"))

    summary_path = os.path.join("Results", "pareto_summary.txt")
    result_path = os.path.join("Results", "pareto_summary.json")

    with open(summary_path, "w") as f:
        f.write(str(summary))

    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)