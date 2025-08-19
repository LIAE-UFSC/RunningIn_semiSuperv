"""
Optuna Hyperparameter Optimization for RunIn Semi-Supervised Learning

Database Configuration:
1. SQLite (default): Set USE_POSTGRES = False
   - No additional setup required
   - Files saved to Results/ directory

2. PostgreSQL: Set USE_POSTGRES = True
   - Requires PostgreSQL server setup
   - Install dependencies: pip install psycopg2-binary
   - Update POSTGRES_CONFIG with your database credentials
   
PostgreSQL Setup:
1. Install PostgreSQL server
2. Install Python PostgreSQL adapter:
   ```bash
   pip install psycopg2-binary
   # OR if you have compilation issues:
   pip install psycopg2
   ```
3. Create database and user:
   ```sql
   CREATE DATABASE optuna_db;
   CREATE USER optuna_user WITH PASSWORD 'optuna_password';
   GRANT ALL PRIVILEGES ON DATABASE optuna_db TO optuna_user;
   ```
4. Update POSTGRES_CONFIG dictionary below

Common Issues:
- "Failed to import DB access module": Install psycopg2-binary
- Connection refused: Check PostgreSQL server is running
- Authentication failed: Verify username/password in POSTGRES_CONFIG
"""

from utils.optimizator import OptimizationRunIn
import logging
import optuna
from scipy.stats import gmean
import sys
import warnings
import multiprocessing as mp
import time

warnings.filterwarnings("ignore", category=UserWarning)

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Optimization for RunIn Semi-Supervised Learning")
    parser.add_argument("--compressor_model", type=str, default="a", choices=["a", "b", "all"],
                        help="Compressor model to use: 'a', 'b', or 'all'")
    parser.add_argument("--n_processes", type=int, default=4, help="Number of parallel processes for optimization")
    parser.add_argument("--max_init_samples", type=int, default=180, help="Maximum total window size for optimization")
    parser.add_argument("--auto_broadcast", action="store_true", help="Automatically start Optuna dashboard")
    parser.add_argument("--pareto_stop", type=int, default=-1, help="Number of trials to stop after not advancing the Pareto front", required=True)
    parser.add_argument("--max_iter", type=int, default=100000, help="Maximum iterations for optimization")
    parser.add_argument("--classifiers", nargs="+", default = ["all"], help="List of classifiers to optimize")
    return parser.parse_args()

# Database configuration
USE_POSTGRES = True  # Set to True to use PostgreSQL, False for SQLite
POSTGRES_CONFIG = {
    'host': '100.107.250.125',
    'port': 5432,
    'database': 'optuna_db',
    'user': 'optuna_user',
    'password': 'optuna_password'
}

if __name__ == "__main__":
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    args = parse_args()
    compressor_model = args.compressor_model
    n_processes = args.n_processes
    max_init_samples = args.max_init_samples
    auto_broadcast = args.auto_broadcast
    pareto_stop = args.pareto_stop
    max_iter = args.max_iter
    classifiers = args.classifiers

    if classifiers == ["all"]:
        classifiers = OptimizationRunIn.supported_classifiers

    for classifier_type in classifiers:

            # Create an instance of the optimization class
            optimizer = OptimizationRunIn(classifier=classifier_type, compressor_model=compressor_model, use_postgres=True)

            optimizer.postgres_config = POSTGRES_CONFIG  # Set the PostgreSQL config
            
            start_time = time.time()

            study_monitor = optimizer.init_study()

            last_pareto = study_monitor.best_trials[-1].number if study_monitor.best_trials else 0

            n_done = len(study_monitor.trials)

            if (n_done - last_pareto) >= pareto_stop:
                print(f"Already reached Pareto front stop condition for {classifier_type}. No further optimization needed.")
                continue
            elif n_done >= max_iter:
                print(f"Already reached maximum iterations ({max_iter}) for {classifier_type}. No further optimization needed.")
                continue
            else:
                print(f"Running iteration of optimizer, aiming for {pareto_stop} tests after advancing.")

            while ((n_done - last_pareto) < pareto_stop) and (n_done < max_iter):
                if (n_done + pareto_stop) >= max_iter:
                    n_tests = max_iter - n_done
                    print(f"Approaching maximum iterations ({max_iter}). Running only {n_tests} more trials.")
                else:
                    n_tests = pareto_stop - (n_done - last_pareto)
                    print(f"Last Pareto front advance has been {n_done - last_pareto} trials ago. Test again after {n_tests} trials.")

                optimizer.multithreading_optimization(n_trials=n_tests, n_jobs=1, n_processes=n_processes, connect_db_allproc=True, method = "fork")

                last_pareto = study_monitor.best_trials[-1].number if study_monitor.best_trials else 0
                n_done = len(study_monitor.trials)

            elapsed = time.time() - start_time
            print(f"Tests finished for classifier {classifier_type}. Elapsed time: {elapsed:.2f} seconds")