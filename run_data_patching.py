# nohup python3 run_data_patching.py --config_path configs/config.yaml > 2>&1 &


from dataset.data_utils import data_patching
import argparse
import logging
import os
import sys

def setup_logger(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Run data patching process.")
    parser.add_argument('--config_path', type=str, required=True, help='Path to the YAML configuration file.')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save the log file.')

    args = parser.parse_args()

    # Prépare le dossier de logs
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, 'data_patching.log')
    setup_logger(log_file)

    logging.info("Starting data patching process...")
    try:
        data_patching(args.config_path)
        logging.info("Data patching completed successfully.")
    except Exception as e:
        logging.exception(f"An error occurred during data patching: {e}")

if __name__ == '__main__':
    main()

