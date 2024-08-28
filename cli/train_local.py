import argparse
from typing import List, Tuple, Dict, Any, Optional

import yaml

import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.optim import Optimizer, Adam

from models.operators import GlobalOperator, LocalOperator
from era5.wind.datasets import Wind2dERA5
from common.training import CheckpointLoader
from workers.trainer import LocalOperatorTrainer


def main(config: Dict[str, Any]) -> None:
    """
    Main function to train Global Operator.

    Parameters:
        config (Dict[str, Any]): Configuration dictionary.
    """

    # Parse CLI arguments:
    device: torch.device                = torch.device(config['device'])
    dataroot: str                       = str(config['dataset']['root'])
    pressure_level: str                 = int(config['dataset']['pressure_level'])
    global_latitude: Tuple[float, float] = tuple(config['dataset']['global_latitude'])
    global_longitude: Tuple[float, float] = tuple(config['dataset']['global_longitude'])
    global_resolution: Tuple[int, int]  = tuple(config['dataset']['global_resolution'])
    local_latitude: Tuple[float, float] = tuple(config['dataset']['local_latitude'])
    local_longitude: Tuple[float, float] = tuple(config['dataset']['local_longitude'])
    local_resolution: Tuple[int, int]   = tuple(config['dataset']['local_resolution'])
    train_fromdate: str                 = str(config['dataset']['train_fromdate'])
    train_todate: str                   = str(config['dataset']['train_todate'])
    val_fromdate: str                   = str(config['dataset']['val_fromdate'])
    val_todate: str                     = str(config['dataset']['val_todate'])
    time_resolution: int                = int(config['dataset']['time_resolution'])
    bundle_size: int                    = int(config['dataset']['bundle_size'])
    input_size: int                     = int(config['dataset']['input_size'])

    from_checkpoint: Optional[str]      = config['global_architecture']['from_checkpoint']
    global_checkpoint: str              = str(config['local_architecture']['global_checkpoint'])
    block_size: int                     = int(config['local_architecture']['block_size'])
    patch_size: tuple                   = tuple(config['local_architecture']['patch_size'])
    n_attention_heads: int              = int(config['local_architecture']['n_attention_heads'])

    noise_level: float                  = float(config['training']['noise_level'])
    train_batch_size: int               = int(config['training']['train_batch_size'])
    val_batch_size: int                 = int(config['training']['val_batch_size'])
    learning_rate: float                = float(config['training']['learning_rate'])
    n_epochs: int                       = int(config['training']['n_epochs'])
    patience: int                       = int(config['training']['patience'])
    tolerance: int                      = float(config['training']['tolerance'])
    save_frequency: int                 = int(config['training']['save_frequency'])

    # Load global operator
    global_loader = CheckpointLoader(checkpoint_path=global_checkpoint)
    global_operator: GlobalOperator
    global_operator, _ = global_loader.load(scope=globals())

    # Instatiate the training datasets
    train_dataset = Wind2dERA5(
        dataroot=dataroot,
        pressure_level=pressure_level,
        fromdate=train_fromdate,
        todate=train_todate,
        global_latitude=global_latitude,
        global_longitude=global_longitude,
        global_resolution=global_resolution,
        local_latitude=local_latitude,
        local_longitude=local_longitude,
        local_resolution=local_resolution,
        time_resolution=time_resolution,
        bundle_size=bundle_size,
        input_size=input_size,
    )
    val_dataset = Wind2dERA5(
        dataroot=dataroot,
        pressure_level=pressure_level,
        fromdate=val_fromdate,
        todate=val_todate,
        global_latitude=global_latitude,
        global_longitude=global_longitude,
        global_resolution=global_resolution,
        local_latitude=local_latitude,
        local_longitude=local_longitude,
        local_resolution=local_resolution,
        time_resolution=time_resolution,
        bundle_size=bundle_size,
        input_size=input_size,
    )

    # Load local operator
    if from_checkpoint is not None:
        local_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
        local_operator: LocalOperator; local_optimizer: Optimizer
        local_operator, local_optimizer = local_loader.load(scope=globals())
    else:
        local_operator = LocalOperator(
            in_channels=global_operator.in_channels, 
            embedding_dim=global_operator.embedding_dim,
            in_timesteps=train_dataset.in_timesteps, 
            out_timesteps=train_dataset.out_timesteps,
            n_layers=global_operator.n_layers,
            spatial_resolution=train_dataset.local_resolution,
            block_size=block_size, 
            patch_size=patch_size,
            n_attention_heads=n_attention_heads,
        )
        local_optimizer = Adam(params=local_operator.parameters(), lr=learning_rate)
    
    # Load local trainer
    trainer = LocalOperatorTrainer(
        local_operator=local_operator,
        global_operator=global_operator, 
        optimizer=local_optimizer,
        noise_level=noise_level,
        train_dataset=train_dataset, val_dataset=val_dataset,
        train_batch_size=train_batch_size, val_batch_size=val_batch_size,
        device=device,
    )
    trainer.train(
        n_epochs=n_epochs, patience=patience,
        tolerance=tolerance, checkpoint_path=f'.checkpoints/local/{pressure_level}',
        save_frequency=save_frequency,
    )


if __name__ == "__main__":

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Train the Local Operator')
    parser.add_argument('--config', type=str, required=True, help='Configuration file name.')

    args: argparse.Namespace = parser.parse_args()
    
    # Load the configuration
    with open(file=args.config, mode='r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Run the main function with the configuration
    main(config)



