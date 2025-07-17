# nohup python3 run_train.py --config configs/config.yaml --RGB_HSI RGB --patch_size 300 --gpu 3 --log_dir logs --log_file training_300.log > /dev/null 2>&1 &


import argparse
import os
import torch
import logging
import sys

from torch.utils.data import DataLoader
from dataset.dataloader import HSIDataset
from src.utils import Path_csvs, load_config
from src.train import train
from src.test import test


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
    # 1. Parser les arguments
    parser = argparse.ArgumentParser(description="Run Training Script")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to YAML config file')
    parser.add_argument('--RGB_HSI', type=str, default='HSI', help='Training on HSI or RGB')
    parser.add_argument('--patch_size', type=int, choices=[100, 200, 300], required=True, help='Patch size to use')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save the log file')
    parser.add_argument('--log_file', type=str, default='training.log', help='Name of log file')
    args = parser.parse_args()

    # 2. Setup logging
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, args.log_file)
    setup_logger(log_file)

    logging.info(f"Starting training with config: {args.config}, patch size: {args.patch_size}, GPU: {args.gpu}")

    try:
        # 3. Setup device
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")

        # 4. Load config and paths
        config = load_config(path=args.config)
        csvs = Path_csvs(config_path=args.config, RGB_HSI=args.RGB_HSI, Patch=args.patch_size)

        # 5. Load datasets
        train_dataset = HSIDataset(csvs['train'])
        val_dataset = HSIDataset(csvs['val'])
        test_dataset = HSIDataset(csvs['test'])

        train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"],
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"],
                                shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"],
                                 shuffle=False, num_workers=4, pin_memory=True)

        model_save_name = csvs['name']

        # 6. Train & Test
        train(config, train_loader, val_loader, model_save_name, device, RGB_HSI=args.RGB_HSI)
        test(config, test_loader, model_save_name, device, checkpoint_type="best", RGB_HSI=args.RGB_HSI)

        logging.info("Training and testing completed successfully.")

    except Exception as e:
        logging.exception(f"An error occurred during training or testing: {e}")


if __name__ == '__main__':
    main()
