import argparse
from typing import Tuple, Dict, Any, Optional

import yaml

import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.optim import Optimizer, Adam

from models.operators import GlobalOperator
from era5.wind.datasets import Wind2dERA5
from common.training import CheckpointLoader
from workers.trainer import GlobalOperatorTrainer


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
    train_fromdate: str                 = str(config['dataset']['train_fromdate'])
    train_todate: str                   = str(config['dataset']['train_todate'])
    val_fromdate: str                   = str(config['dataset']['val_fromdate'])
    val_todate: str                     = str(config['dataset']['val_todate'])
    time_resolution: int                = int(config['dataset']['time_resolution'])
    bundle_size: int                    = int(config['dataset']['bundle_size'])
    input_size: int                     = int(config['dataset']['input_size'])

    in_channels: int                    = int(config['global_architecture']['in_channels'])
    embedding_dim: int                  = int(config['global_architecture']['embedding_dim'])
    n_layers: int                       = int(config['global_architecture']['n_layers'])
    block_size: int                     = int(config['global_architecture']['block_size'])
    patch_size: int                     = tuple(config['global_architecture']['patch_size'])
    dropout_rate: float                 = float(config['global_architecture']['dropout_rate'])
    from_checkpoint: Optional[str]      = config['global_architecture']['from_checkpoint']
    
    noise_level: float                  = float(config['training']['noise_level'])
    train_batch_size: int               = int(config['training']['train_batch_size'])
    val_batch_size: int                 = int(config['training']['val_batch_size'])
    learning_rate: float                = float(config['training']['learning_rate'])
    n_epochs: int                       = int(config['training']['n_epochs'])
    patience: int                       = int(config['training']['patience'])
    tolerance: int                      = float(config['training']['tolerance'])
    multistep_training: bool            = bool(config['training']['multistep_training'])
    save_frequency: int                 = int(config['training']['save_frequency'])

    # Instatiate the training datasets
    train_dataset = Wind2dERA5(
        dataroot=dataroot,
        pressure_level=pressure_level,
        fromdate=train_fromdate,
        todate=train_todate,
        global_latitude=global_latitude,
        global_longitude=global_longitude,
        global_resolution=global_resolution,
        local_latitude=None,
        local_longitude=None,
        local_resolution=None,
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
        local_latitude=None,
        local_longitude=None,
        local_resolution=None,
        time_resolution=time_resolution,
        bundle_size=bundle_size,
        input_size=input_size,
    )

    # Load global operator
    if from_checkpoint is not None:
        checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint + f'/{pressure_level}')
        operator: GlobalOperator; optimizer: Optimizer
        operator, optimizer = checkpoint_loader.load(scope=globals())
    
    else:
        if multistep_training:
            model_out_timesteps: int = train_dataset.timesteps_per_day
        else:
            model_out_timesteps: int = train_dataset.out_timesteps

        operator = GlobalOperator(
            in_channels=in_channels, 
            embedding_dim=embedding_dim,
            in_timesteps=train_dataset.in_timesteps, 
            out_timesteps=model_out_timesteps,
            n_layers=n_layers,
            spatial_resolution=global_resolution,
            block_size=block_size, 
            patch_size=patch_size,
            dropout_rate=dropout_rate,
        )
        optimizer = Adam(params=operator.parameters(), lr=learning_rate)

    # Load global trainer    
    trainer = GlobalOperatorTrainer(
        global_operator=operator, optimizer=optimizer,
        noise_level=noise_level,
        train_dataset=train_dataset, val_dataset=val_dataset,
        train_batch_size=train_batch_size, val_batch_size=val_batch_size,
        multistep_training=multistep_training,
        device=device,
    )
    trainer.train(
        n_epochs=n_epochs, patience=patience,
        tolerance=tolerance, checkpoint_path=f'.checkpoints/global/{pressure_level}',
        save_frequency=save_frequency,
    )


if __name__ == "__main__":

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Train the Global Operator')
    parser.add_argument('--config', type=str, required=True, help='Configuration file name.')

    args: argparse.Namespace = parser.parse_args()
    
    # Load the configuration
    with open(file=args.config, mode='r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Run the main function with the configuration
    main(config)




