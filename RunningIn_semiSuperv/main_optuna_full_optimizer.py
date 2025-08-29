"""
Optuna Hyperparameter Optimization for RunIn Semi-Supervised Learning

Database Configuration:
1. SQLite (default): Set USE_POSTGRES = False
   - No additional setup required
   - Files saved to Results/ directory

2. PostgreSQL: Set USE_POSTGRES = True
   - Requires PostgreSQL server setup
   - Install dependencies: pip install psycopg2-binary
   - Configuration loaded from postgres_config.json or defaults used
   
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
4. Configuration will be loaded automatically from postgres_config.json

Common Issues:
- "Failed to import DB access module": Install psycopg2-binary
- Connection refused: Check PostgreSQL server is running
- Authentication failed: Check configuration in postgres_config.json
"""

from utils import RunInSemiSupervised
from utils.optimizer import OptimizationRunIn
import logging
import optuna
from scipy.stats import gmean
from pathlib import Path
import sys
import numpy as np
import warnings
import multiprocessing as mp
import time
import subprocess
import signal
import os

warnings.filterwarnings("ignore", category=UserWarning)

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Optimization for RunIn Semi-Supervised Learning")
    parser.add_argument("--compressor_model", type=str, default="a", choices=["a", "b", "all"],
                        help="Compressor model to use: 'a', 'b', or 'all'")
    parser.add_argument("--n_processes", type=int, default=4, help="Number of parallel processes for optimization")
    parser.add_argument("--n_tests", type=int, default=1000, help="Total number of optimization trials")
    parser.add_argument("--max_init_samples", type=int, default=180, help="Maximum total window size for optimization")
    parser.add_argument("--auto_broadcast", action="store_true", help="Automatically start Optuna dashboard")
    return parser.parse_args()

args = parse_args()
compressor_model = args.compressor_model
n_processes = args.n_processes
n_tests = args.n_tests
max_init_samples = args.max_init_samples
auto_broadcast = args.auto_broadcast

# Database configuration
USE_POSTGRES = True  # Set to True to use PostgreSQL, False for SQLite

def MCC_score_from_confusion_matrix(cm):
    """
    Calculate Matthews Correlation Coefficient (MCC) from confusion matrix.
    
    Parameters:
    cm (array-like): Confusion matrix
    
    Returns:
    float: MCC score
    """
    tn, fp, fn, tp = cm.ravel()
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return numerator / denominator if denominator != 0 else 0



def start_optuna_dashboard(storage_name, port=8080):
    """
    Start Optuna dashboard in a subprocess.
    
    Args:
        storage_name (str): Path to the database file
        port (int): Port number for the dashboard
    
    Returns:
        subprocess.Popen: Dashboard process object
    """
    try:
        print(f"Starting Optuna dashboard at http://127.0.0.1:{port}")
        
        # Handle different operating systems for process group creation
        if os.name == 'nt':  # Windows
            dashboard_process = subprocess.Popen(
                ["optuna-dashboard", storage_name, "--host 0.0.0.0 --port", str(port), ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:  # Unix-like systems (Linux, macOS)
            dashboard_process = subprocess.Popen(
                ["optuna-dashboard", storage_name, "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid  # Create new process group for easier termination
            )
        return dashboard_process
    except FileNotFoundError:
        print("Warning: optuna-dashboard not found. Please install with: pip install optuna-dashboard")
        return None
    except Exception as e:
        print(f"Warning: Could not start Optuna dashboard: {e}")
        return None


def stop_optuna_dashboard(dashboard_process):
    """
    Stop the Optuna dashboard process.
    
    Args:
        dashboard_process (subprocess.Popen): Dashboard process object
    """
    if dashboard_process is not None:
        try:
            if os.name == 'nt':  # Windows
                # On Windows, terminate the process directly
                dashboard_process.terminate()
                dashboard_process.wait(timeout=5)  # Wait up to 5 seconds for graceful shutdown
                print("Optuna dashboard stopped successfully")
            else:  # Unix-like systems (Linux, macOS)
                # Send SIGTERM to the process group
                os.killpg(os.getpgid(dashboard_process.pid), signal.SIGTERM)
                dashboard_process.wait(timeout=5)  # Wait up to 5 seconds for graceful shutdown
                print("Optuna dashboard stopped successfully")
        except subprocess.TimeoutExpired:
            # Force kill if graceful shutdown failed
            if os.name == 'nt':  # Windows
                dashboard_process.kill()
                print("Optuna dashboard force killed")
            else:  # Unix-like systems
                os.killpg(os.getpgid(dashboard_process.pid), signal.SIGKILL)
                print("Optuna dashboard force killed")
        except Exception as e:
            print(f"Warning: Could not stop dashboard process: {e}")


if __name__ == "__main__":
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    for classifier_type in OptimizationRunIn.supported_classifiers:
        dashboard_process = None
        processes = []
        
        try:
            # Create an instance of the optimization class
            optimizer = OptimizationRunIn(
                classifier=classifier_type, 
                compressor_model=compressor_model,
                max_init_samples=max_init_samples,
                use_postgres=USE_POSTGRES
            )
            
            # Get database URL (SQLite or PostgreSQL)
            storage_name = optimizer.get_database_url()
            
            # Create Optuna study name
            study_name = f"RunIn_{classifier_type}_{compressor_model}"
            
            # Configure storage with timeout settings
            try:
                if USE_POSTGRES:
                    # PostgreSQL storage with connection pooling and timeout
                    storage = optuna.storages.RDBStorage(
                        url=storage_name,
                        engine_kwargs={
                            "pool_size": 20,
                            "max_overflow": 0,
                            "pool_pre_ping": True,
                            "pool_recycle": 300,
                            "connect_args": {"connect_timeout": 60}
                        }
                    )
                else:
                    # SQLite storage with timeout
                    storage = optuna.storages.RDBStorage(
                        url=storage_name, 
                        engine_kwargs={"connect_args": {"timeout": 100}}
                    )
            except Exception as e:
                if "Failed to import DB access module" in str(e):
                    raise ImportError(
                        f"Database connection failed for {classifier_type}. "
                        f"If using PostgreSQL, install psycopg2-binary:\n"
                        f"pip install psycopg2-binary\n"
                        f"Original error: {e}"
                    )
                else:
                    raise e
            
            # Create Optuna study
            study = optuna.create_study(
                directions=["maximize", "maximize"], 
                study_name=study_name, 
                storage=storage, 
                load_if_exists=True
            )
            
            if auto_broadcast:        
                # Start Optuna dashboard
                dashboard_port = 8081
                dashboard_process = start_optuna_dashboard(storage_name, port=dashboard_port)
            else:
                dashboard_process = None
            
            start_time = time.time()

            # Calculate number of studies per process
            N_studies_left = n_tests
            n_studies_process = round(N_studies_left / n_processes)

            for i in range(n_processes):  # Number of optimization trials
                if i == n_processes - 1: # Last process takes remaining trials
                    n_studies_process = N_studies_left
                else:
                    N_studies_left -= n_studies_process
                p = mp.Process(target=study.optimize, args=(optimizer.objective,), kwargs={'n_trials': n_studies_process, 'n_jobs': 1})
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            elapsed = time.time() - start_time
            print(f"Tests finished for classifier {classifier_type}. Elapsed time: {elapsed:.2f} seconds")
            
        except KeyboardInterrupt:
            print(f"\nKeyboard interrupt received. Cleaning up processes for {classifier_type}...")
            # Terminate all optimization processes
            for p in processes:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                        p.kill()  # Force kill if terminate didn't work
            raise  # Re-raise to exit the main loop
            
        except Exception as e:
            print(f"Error occurred during optimization for {classifier_type}: {e}")
            # Terminate all optimization processes
            for p in processes:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                        p.kill()  # Force kill if terminate didn't work
            
        finally:
            # Always stop the dashboard, regardless of success or failure
            if dashboard_process is not None:
                stop_optuna_dashboard(dashboard_process)