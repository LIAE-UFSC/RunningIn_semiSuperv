
import logging
import optuna
import sys
import warnings
import multiprocessing as mp
import time
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT # <-- ADD THIS LINE
import argparse
import os
from utils.optimizer import OptimizationRunIn
import getpass

# Database configuration
# NOTE: Update 'admin_password' with your actual PostgreSQL admin password
POSTGRES_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'optuna_test',
    'user': 'optuna_test_user',
    'password': 'test_password',
    'admin_user': 'postgres',
}

n_proc = [1,2,3,4,5,10]

def parse_args():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Optimization for RunIn Semi-Supervised Learning")
    parser.add_argument("--compressor_model", type=str, default="a", choices=["a", "b", "all"],
                        help="Compressor model to use: 'a', 'b', or 'all'")
    parser.add_argument("--n_tests", type=int, default=100, help="Total number of optimization trials")
    parser.add_argument("--classifiers", nargs="+", default = ["all"], help="List of classifiers to optimize")
    return parser.parse_args()

def setup_test_database():
    """Create test database and user with full access."""
    try:
        # Connect as admin user to create database and user
        admin_con = psycopg2.connect(
            dbname='postgres',
            user=POSTGRES_CONFIG['admin_user'], 
            host=POSTGRES_CONFIG['host'],
            password=POSTGRES_CONFIG['admin_password']
        )
        admin_con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        admin_cur = admin_con.cursor()
        
        # Create user if it doesn't exist
        admin_cur.execute(sql.SQL("""
            SELECT 1 FROM pg_roles WHERE rolname = %s
        """), [POSTGRES_CONFIG['user']])
        
        if not admin_cur.fetchone():
            admin_cur.execute(sql.SQL("""
                CREATE USER {} WITH PASSWORD %s
            """).format(sql.Identifier(POSTGRES_CONFIG['user'])), 
            [POSTGRES_CONFIG['password']])
            print(f"Created user: {POSTGRES_CONFIG['user']}")
        else:
            print(f"User {POSTGRES_CONFIG['user']} already exists")
        
        # Create database if it doesn't exist
        admin_cur.execute(sql.SQL("""
            SELECT 1 FROM pg_database WHERE datname = %s
        """), [POSTGRES_CONFIG['database']])
        
        if not admin_cur.fetchone():
            admin_cur.execute(sql.SQL("CREATE DATABASE {}").format(
                sql.Identifier(POSTGRES_CONFIG['database'])
            ))
            print(f"Created database: {POSTGRES_CONFIG['database']}")
        else:
            print(f"Database {POSTGRES_CONFIG['database']} already exists")
        
        # Grant all privileges on database to user
        admin_cur.execute(sql.SQL("""
            GRANT ALL PRIVILEGES ON DATABASE {} TO {}
        """).format(
            sql.Identifier(POSTGRES_CONFIG['database']),
            sql.Identifier(POSTGRES_CONFIG['user'])
        ))
        
        admin_cur.close()
        admin_con.close()
        
        # Connect to the test database to grant schema permissions
        test_con = psycopg2.connect(
            dbname=POSTGRES_CONFIG['database'],
            user=POSTGRES_CONFIG['admin_user'],
            host=POSTGRES_CONFIG['host'],
            password=POSTGRES_CONFIG['admin_password']
        )
        test_con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        test_cur = test_con.cursor()
        
        # Grant all privileges on the public schema and all objects within it
        test_cur.execute(sql.SQL("""
            GRANT ALL ON SCHEMA public TO {}
        """).format(sql.Identifier(POSTGRES_CONFIG['user'])))
        
        test_cur.execute(sql.SQL("""
            GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO {}
        """).format(sql.Identifier(POSTGRES_CONFIG['user'])))
        
        test_cur.execute(sql.SQL("""
            GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO {}
        """).format(sql.Identifier(POSTGRES_CONFIG['user'])))
        
        test_cur.execute(sql.SQL("""
            GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO {}
        """).format(sql.Identifier(POSTGRES_CONFIG['user'])))
        
        # Grant privileges on future objects in the public schema
        test_cur.execute(sql.SQL("""
            ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO {}
        """).format(sql.Identifier(POSTGRES_CONFIG['user'])))
        
        test_cur.execute(sql.SQL("""
            ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO {}
        """).format(sql.Identifier(POSTGRES_CONFIG['user'])))
        
        test_cur.execute(sql.SQL("""
            ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO {}
        """).format(sql.Identifier(POSTGRES_CONFIG['user'])))
        
        test_cur.close()
        test_con.close()
        
        print(f"Setup complete: database '{POSTGRES_CONFIG['database']}' and user '{POSTGRES_CONFIG['user']}' with full schema permissions")
        
    except Exception as e:
        print(f"Error setting up test database: {e}")
        raise

def cleanup_test_database():
    """Clean up test database and user."""
    try:
        # Connect as admin user to drop database and user
        admin_con = psycopg2.connect(
            dbname='postgres',
            user=POSTGRES_CONFIG['admin_user'],
            host=POSTGRES_CONFIG['host'],
            password=POSTGRES_CONFIG['admin_password']
        )
        admin_con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        admin_cur = admin_con.cursor()
        
        # Terminate all connections to the database before dropping
        admin_cur.execute(sql.SQL("""
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = %s AND pid <> pg_backend_pid();
        """), [POSTGRES_CONFIG['database']])
        
        # Drop database if it exists
        admin_cur.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(
            sql.Identifier(POSTGRES_CONFIG['database'])
        ))
        print(f"Dropped database: {POSTGRES_CONFIG['database']}")
        
        # Drop user if it exists
        admin_cur.execute(sql.SQL("DROP USER IF EXISTS {}").format(
            sql.Identifier(POSTGRES_CONFIG['user'])
        ))
        print(f"Dropped user: {POSTGRES_CONFIG['user']}")
        
        admin_cur.close()
        admin_con.close()
        
    except Exception as e:
        print(f"Error cleaning up test database: {e}")

def append_result_to_csv(result_data, file_path):
    """
    Append a single result to the CSV file.
    
    Parameters:
    result_data (dict): Dictionary containing the result data
    file_path (str): Path to the CSV file
    """
    try:
        # Create a DataFrame with the single result
        result_df = pd.DataFrame([result_data])
        # Ensure the directory exists
        results_dir = os.path.dirname(file_path)
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
        # Append to CSV file
        result_df.to_csv(file_path, mode='a', header=False, index=False)
        print(f"Appended result for {result_data.get('type', 'unknown')} classifier to {file_path}")
    except Exception as e:
        print(f"Error appending result to CSV: {e}")


if __name__ == "__main__":

    args = parse_args()
    # Prompt for PostgreSQL admin password if not set in config
    if 'admin_password' not in POSTGRES_CONFIG or not POSTGRES_CONFIG['admin_password']:
        POSTGRES_CONFIG['admin_password'] = getpass.getpass("Enter PostgreSQL admin password: ")

    warnings.filterwarnings("ignore", category=UserWarning)

    compressor_model = args.compressor_model
    n_tests = args.n_tests
    classifiers = args.classifiers

    if classifiers == ["all"]:
        classifiers = OptimizationRunIn.supported_classifiers

    max_init_samples = 180

    # Setup test database and user
    setup_test_database()
    
    # Initialize CSV file for results
    results_file = f"optuna_results_{compressor_model}.csv"
    csv_columns = ["n_proc", "n_jobs", "elapsed_time", "type"]

    # Create CSV file with headers if it doesn't exist
    try:
        pd.read_csv(results_file)
        print(f"Results file {results_file} already exists, will append to it.")
    except FileNotFoundError:
        # Create empty CSV with headers
        empty_df = pd.DataFrame(columns=csv_columns)
        empty_df.to_csv(results_file, index=False)
        print(f"Created new results file: {results_file}")

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    try:
        for classifier_type in classifiers:

            # Create an instance of the optimization class
            optimizer = OptimizationRunIn(classifier=classifier_type, compressor_model=compressor_model, use_postgres=True)

            optimizer.postgres_config = POSTGRES_CONFIG  # Set the PostgreSQL config

            for n_processes in n_proc:

                start_time = time.time()

                optimizer.multithreading_optimization(n_trials=n_tests, n_jobs=1, n_processes=n_processes, connect_db_allproc=True, method = "fork")

                elapsed = time.time() - start_time
                result = {}
                result["n_proc"] = n_processes
                result["n_jobs"] = 1
                result["elapsed_time"] = elapsed
                result["type"] = classifier_type
                
                # Append result to CSV immediately
                append_result_to_csv(result, results_file)

            print(f"Tests finished for classifier {classifier_type}.")
                
            
    
    finally:
        # Clean up test database and user at the end
        cleanup_test_database()
        print(f"All results have been saved to {results_file}")
        print("Optimization completed!")
        