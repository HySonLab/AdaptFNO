from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from functools import cached_property

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from common.training import Accumulator, EarlyStopping, Timer, Logger, CheckpointSaver
from common.losses import VGG16Loss

from models.operators import GlobalOperator, LocalOperator
from era5.wind.datasets import Wind2dERA5


class _BaseOperatorTrainer(ABC):

    def __init__(
        self, 
        optimizer: Optimizer,
        noise_level: float,
        train_dataset: Wind2dERA5,
        val_dataset: Wind2dERA5,
        train_batch_size: int,
        val_batch_size: int,
        multistep_training: bool,
        device: torch.device,
    ):
        self.optimizer: Optimizer = optimizer
        self.noise_level: float = noise_level
        self.train_dataset: Wind2dERA5 = train_dataset
        self.val_dataset: Wind2dERA5 = val_dataset
        self.train_batch_size: int = train_batch_size
        self.val_batch_size: int = val_batch_size
        self.multistep_training: bool = multistep_training
        self.device: torch.device = device

        self.train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=train_batch_size, 
            shuffle=True,
        )
        self.val_dataloader = DataLoader(
            dataset=val_dataset, 
            batch_size=val_batch_size, 
            shuffle=False,
        )
        self.loss_function: nn.Module = nn.MSELoss(reduction='sum').to(self.device)
        # self.loss_function: nn.Module = VGG16Loss(reduction='sum').to(self.device)

    @abstractmethod
    def train(
        self, 
        n_epochs: int,
        patience: int,
        tolerance: float,
        checkpoint_path: Optional[str] = None,
        save_frequency: int = 5,
    ) -> None:
        pass
    
    @abstractmethod
    def evaluate(self) -> Tuple[float, float]:
        pass


class GlobalOperatorTrainer(_BaseOperatorTrainer):

    def __init__(
        self, 
        global_operator: GlobalOperator,
        optimizer: Optimizer,
        noise_level: float,
        train_dataset: Wind2dERA5,
        val_dataset: Wind2dERA5,
        train_batch_size: int,
        val_batch_size: int,
        multistep_training: bool,
        device: torch.device,
    ):
        super().__init__(
            optimizer=optimizer, 
            noise_level=noise_level, 
            train_dataset=train_dataset, val_dataset=val_dataset,
            train_batch_size=train_batch_size, val_batch_size=val_batch_size,
            multistep_training=multistep_training,
            device=device,
        )
        self.global_operator: GlobalOperator = global_operator.to(device=self.device)

    def train(
        self, 
        n_epochs: int,
        patience: int,
        tolerance: float,
        checkpoint_path: Optional[str] = None,
        save_frequency: int = 5,
    ) -> None:
        
        train_metrics = Accumulator()
        early_stopping = EarlyStopping(patience, tolerance)
        timer = Timer()
        logger = Logger()
        checkpoint_saver = CheckpointSaver(
            model=self.global_operator,
            optimizer=self.optimizer,
            dirpath=checkpoint_path,
        )
        self.global_operator.train()
        
        # loop through each epoch
        for epoch in range(1, n_epochs + 1):
            timer.start_epoch(epoch)
            # Loop through each batch
            for batch, (batch_input, batch_groundtruth) in enumerate(self.train_dataloader, start=1):
                timer.start_batch(epoch, batch)
                assert batch_input.ndim == batch_groundtruth.ndim == 5
                batch_size, window_size, u_dim, x_res, y_res = batch_input.shape
                # Move to the selected device
                batch_input: torch.Tensor = batch_input.to(device=self.device)
                batch_groundtruth: torch.Tensor = batch_groundtruth.to(device=self.device)
                # Reset gradients
                self.optimizer.zero_grad()
                
                # Mutil-step Training
                prediction_steps: int = self.train_dataset.bundle_size
                timesteps_per_prediction: int = self.train_dataset.timesteps_per_day    # operator was set such that 1 prediction = 1 day
                # Prepare input
                batch_input += torch.randn_like(input=batch_input, device=self.device) * batch_input.std() * self.noise_level
                # Stepping
                total_mse_loss = 0.
                n_elems: int = 0
                for step in range(prediction_steps):
                    step_slice = slice(step * timesteps_per_prediction, (step + 1) * timesteps_per_prediction)
                    step_groundtruth: torch.Tensor = batch_groundtruth[:, step_slice, ...]
                    # Forward propagation
                    batch_prediction, *_ = self.global_operator(input=batch_input)
                    # Accumulate loss
                    total_mse_loss += self.loss_function(input=batch_prediction, target=step_groundtruth)
                    n_elems += batch_prediction.numel()
                    # Prepare input for the next step
                    full_input: torch.Tensor = torch.cat(tensors=[batch_input, batch_prediction], dim=1)
                    batch_input = full_input[:, -batch_input.shape[1]:, ...]

                # Backpropagation
                mean_mse_loss: torch.Tensor = total_mse_loss / n_elems
                mean_mse_loss.backward()
                self.optimizer.step()

                # Accumulate the metrics
                train_metrics.add(total_mse=total_mse_loss.item(), n_elems=n_elems)
                timer.end_batch(epoch=epoch)
                # Log
                mean_train_mse: float = train_metrics['total_mse'] / train_metrics['n_elems']
                logger.log(
                    epoch=epoch, n_epochs=n_epochs, 
                    batch=batch, n_batches=len(self.train_dataloader), 
                    took=timer.time_batch(epoch, batch), 
                    train_rmse=mean_train_mse ** 0.5, train_mse=mean_train_mse, 
                )
        
            # Ragularly save checkpoint
            if checkpoint_path is not None and epoch % save_frequency == 0:
                checkpoint_saver.save(
                    model_states=self.global_operator.state_dict(), 
                    optimizer_states=self.optimizer.state_dict(),
                    filename=f'epoch{epoch}.pt',
                )
            
            # Reset metric records for next epoch
            train_metrics.reset()
            # Evaluate
            val_rmse, val_mse = self.evaluate()
            timer.end_epoch(epoch)
            # Log
            logger.log(
                epoch=epoch, n_epochs=n_epochs, 
                took=timer.time_epoch(epoch), 
                val_rmse=val_rmse, val_mse=val_mse, 
            )
            print('=' * 20)

            # Check early-stopping
            early_stopping(value=val_mse)
            if early_stopping:
                print('Early Stopped')
                break

        # Always save last checkpoint
        if checkpoint_path:
            checkpoint_saver.save(
                model_states=self.global_operator.state_dict(), 
                optimizer_states=self.optimizer.state_dict(),
                filename=f'epoch{epoch}.pt',
            )

    def evaluate(self) -> Tuple[float, float]:
        val_metrics = Accumulator()
        self.global_operator.eval()
        with torch.no_grad():
            # Loop through each batch
            for batch_input, batch_groundtruth in self.val_dataloader:
                assert batch_input.ndim == 5
                batch_size, window_size, u_dim, x_res, y_res = batch_input.shape
                # Move to the selected device
                batch_input: torch.Tensor = batch_input.to(device=self.device)
                batch_groundtruth: torch.Tensor = batch_groundtruth.to(device=self.device)

                # Mutil-step Prediction
                prediction_steps: int = self.train_dataset.bundle_size
                timesteps_per_prediction: int = self.train_dataset.timesteps_per_day    # operator was set such that 1 prediction = 1 day
                # Stepping
                total_mse_loss = 0.
                n_elems: int = 0
                for step in range(prediction_steps):
                    step_slice = slice(step * timesteps_per_prediction, (step + 1) * timesteps_per_prediction)
                    step_groundtruth: torch.Tensor = batch_groundtruth[:, step_slice, ...]
                    # Forward propagation
                    batch_prediction, *_ = self.global_operator(input=batch_input)
                    # Accumulate loss
                    total_mse_loss += self.loss_function(input=batch_prediction, target=step_groundtruth)
                    n_elems += batch_prediction.numel()
                    # Prepare input for the next step
                    full_input: torch.Tensor = torch.cat(tensors=[batch_input, batch_prediction], dim=1)
                    batch_input = full_input[:, -batch_input.shape[1]:, ...]
                
                # Accumulate the val_metrics
                val_metrics.add(total_mse=total_mse_loss.item(), n_elems=n_elems)

        # Compute the aggregate metrics
        val_mse: float = val_metrics['total_mse'] / val_metrics['n_elems']
        val_rmse: float = val_mse ** 0.5
        return val_rmse, val_mse



class LocalOperatorTrainer(_BaseOperatorTrainer):

    def __init__(
        self, 
        local_operator: LocalOperator,
        global_operator: GlobalOperator,
        optimizer: Optimizer,
        noise_level: float,
        train_dataset: Wind2dERA5,
        val_dataset: Wind2dERA5,
        train_batch_size: int,
        val_batch_size: int,
        device: torch.device,
    ):
        super().__init__(
            optimizer=optimizer, 
            noise_level=noise_level, 
            train_dataset=train_dataset, val_dataset=val_dataset,
            train_batch_size=train_batch_size, val_batch_size=val_batch_size,
            device=device,
        )
        self.local_operator: LocalOperator = local_operator.to(device=self.device)
        self.global_operator: GlobalOperator = global_operator.to(device=self.device)

    def train(
        self, 
        n_epochs: int,
        patience: int,
        tolerance: float,
        checkpoint_path: Optional[str] = None,
        save_frequency: int = 5,
    ) -> None:
        
        train_metrics = Accumulator()
        early_stopping = EarlyStopping(patience, tolerance)
        timer = Timer()
        logger = Logger()
        checkpoint_saver = CheckpointSaver(
            model=self.local_operator,
            optimizer=self.optimizer,
            dirpath=checkpoint_path,
        )
        self.local_operator.train()
        
        # loop through each epoch
        for epoch in range(1, n_epochs + 1):
            timer.start_epoch(epoch)
            # Loop through each batch
            for batch, (
                batch_global_input, _, 
                batch_local_input, batch_local_groundtruth
            ) in enumerate(self.train_dataloader, start=1):
                
                timer.start_batch(epoch, batch)
                assert batch_local_input.ndim == 5
                batch_size, window_size, u_dim, x_res, y_res = batch_local_input.shape
                # Move to the selected device
                batch_global_input: torch.Tensor = batch_global_input.to(device=self.device)
                batch_local_input: torch.Tensor = batch_local_input.to(device=self.device)
                batch_local_groundtruth: torch.Tensor = batch_local_groundtruth.to(device=self.device)
                # Forward propagation
                self.optimizer.zero_grad()
                batch_local_input += (
                    torch.randn_like(input=batch_local_input, device=self.device) * batch_local_input.std() * self.noise_level
                )
                with torch.no_grad():
                    batch_global_contexts: Tuple[torch.Tensor, ...]
                    _, *batch_global_contexts = self.global_operator(input=batch_global_input)

                batch_local_prediction: torch.Tensor = self.local_operator(
                    input=batch_local_input, global_contexts=list(batch_global_contexts),
                )
                # Compute loss
                total_mse_loss: torch.Tensor = self.loss_function(input=batch_local_prediction, target=batch_local_groundtruth)
                mean_mse_loss: torch.Tensor = total_mse_loss / batch_local_prediction.numel()
                # Back propagation
                mean_mse_loss.backward()
                self.optimizer.step()

                # Accumulate the metrics
                train_metrics.add(total_mse=total_mse_loss.item(), n_elems=batch_local_prediction.numel())
                timer.end_batch(epoch=epoch)
                # Log
                mean_train_mse: float = train_metrics['total_mse'] / train_metrics['n_elems']
                logger.log(
                    epoch=epoch, n_epochs=n_epochs, 
                    batch=batch, n_batches=len(self.train_dataloader), 
                    took=timer.time_batch(epoch, batch), 
                    train_rmse=mean_train_mse ** 0.5, train_mse=mean_train_mse, 
                )
        
            # Ragularly save checkpoint
            if checkpoint_path is not None and epoch % save_frequency == 0:
                checkpoint_saver.save(
                    model_states=self.local_operator.state_dict(), 
                    optimizer_states=self.optimizer.state_dict(),
                    filename=f'epoch{epoch}.pt',
                )
            
            # Reset metric records for next epoch
            train_metrics.reset()
            # Evaluate
            val_rmse, val_mse = self.evaluate()
            timer.end_epoch(epoch)
            # Log
            logger.log(
                epoch=epoch, n_epochs=n_epochs, 
                took=timer.time_epoch(epoch), 
                val_rmse=val_rmse, val_mse=val_mse, 
            )
            print('=' * 20)

            # Check early-stopping
            early_stopping(value=val_mse)
            if early_stopping:
                print('Early Stopped')
                break

        # Always save last checkpoint
        if checkpoint_path:
            checkpoint_saver.save(
                model_states=self.local_operator.state_dict(), 
                optimizer_states=self.optimizer.state_dict(),
                filename=f'epoch{epoch}.pt',
            )

    def evaluate(self) -> Tuple[float, float]:
        val_metrics = Accumulator()
        self.global_operator.eval()
        self.local_operator.eval()
        with torch.no_grad():
            # Loop through each batch
            for batch_global_input, _, batch_local_input, batch_local_groundtruth in self.val_dataloader:
                assert batch_local_input.ndim == 5
                batch_size, window_size, u_dim, x_res, y_res = batch_local_input.shape
                # Move to selected device
                batch_global_input: torch.Tensor = batch_global_input.to(device=self.device)
                batch_local_input: torch.Tensor = batch_local_input.to(device=self.device)
                batch_local_groundtruth: torch.Tensor = batch_local_groundtruth.to(device=self.device)
                # Forward propagation
                batch_global_contexts: Tuple[torch.Tensor, ...]
                _, *batch_global_contexts = self.global_operator(input=batch_global_input)
                batch_local_prediction: torch.Tensor = self.local_operator(
                    input=batch_local_input, global_contexts=batch_global_contexts,
                )
                # Compute loss
                total_mse_loss: torch.Tensor = self.loss_function(
                    input=batch_local_prediction, target=batch_local_groundtruth,
                )
                # Accumulate the val_metrics
                val_metrics.add(total_mse=total_mse_loss.item(), n_elems=batch_local_prediction.numel())

        # Compute the aggregate metrics
        val_mse: float = val_metrics['total_mse'] / val_metrics['n_elems']
        val_rmse: float = val_mse ** 0.5
        return val_rmse, val_mse

