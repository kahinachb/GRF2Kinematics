"""
LSTM Training Script for Biomechanical Force-to-Angle Prediction
Loads NPY data, creates sequences, and trains an LSTM model to predict
joint joints from ground reaction forces and moments.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import build_loaders_shared

class ForcesTojointsLSTM(nn.Module):
    """LSTM/BiLSTM + petit MLP pour prédire les angles articulaires à partir des forces."""

    def __init__(self, 
                 input_size: int = 12,
                 hidden_size: int = 128,
                 output_size: int = 12,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 bidirectional: bool = False):
        """
        Args:
            input_size: taille des features d'entrée
            hidden_size: taille des états cachés
            output_size: taille des sorties (angles)
            num_layers: nombre de couches LSTM empilées
            dropout: taux de dropout
            bidirectional: True -> BiLSTM, False -> LSTM
        """
        super().__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers  = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM / BiLSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # Fully connected layers
        fc_in = hidden_size * self.num_directions
        self.fc1 = nn.Linear(fc_in, fc_in // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_in // 2, output_size)

    def forward(self, x):
        """x: [batch_size, seq_len, input_size]"""
        lstm_out, _ = self.lstm(x)  # [B, L, H * num_directions]
        B, L, _ = lstm_out.shape

        out = lstm_out.reshape(-1, self.hidden_size * self.num_directions)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = out.reshape(B, L, self.output_size)
        return out



class Trainer:
    """Trainer class for the LSTM model."""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 learning_rate: float = 0.001,
                 device: str = 'cuda'):
        """
        Initialize the trainer.
        
        Args:
            model: The LSTM model
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (forces, joints) in enumerate(progress_bar):
            forces = forces.to(self.device)
            joints = joints.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(forces)
            
            # Compute loss
            loss = self.criterion(predictions, joints)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model."""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for forces, joints in self.val_loader:
                forces = forces.to(self.device)
                joints = joints.to(self.device)
                
                predictions = self.model(forces)
                loss = self.criterion(predictions, joints)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs: int, save_path: str = './checkpoints_lstm'):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            save_path: Path to save model checkpoints
        """
        Path(save_path).mkdir(exist_ok=True)
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            print(f"Training Loss: {train_loss:.6f}")
            
            # Validate
            val_loss = self.validate()
            if val_loss is not None:
                self.val_losses.append(val_loss)
                print(f"Validation Loss: {val_loss:.6f}")
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'model_hparams': {  # <— add this
                                'input_size': self.model.input_size,
                                'hidden_size': self.model.hidden_size,
                                'output_size': self.model.output_size,
                                'num_layers': self.model.num_layers,
                                'dropout': self.model.lstm.dropout if hasattr(self.model.lstm, 'dropout') else 0.0,
                            },
                        }, Path(save_path) / 'best_model.pth')
                    print("✓ Saved best model")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'model_hparams': {  # <— add this
                                'input_size': self.model.input_size,
                                'hidden_size': self.model.hidden_size,
                                'output_size': self.model.output_size,
                                'num_layers': self.model.num_layers,
                                'dropout': self.model.lstm.dropout if hasattr(self.model.lstm, 'dropout') else 0.0,
                            },
                        }, Path(save_path) / f'checkpoint_epoch_{epoch}.pth')
        
        # Plot training history
        self.plot_training_history()
    
    def plot_training_history(self):
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_history.png')
        # plt.show()



def main():
    parser = argparse.ArgumentParser(description='Train LSTM for force-to-angle prediction')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing processed NPY files')
    parser.add_argument('--sequence_length', type=int, default=10,
                        help='Length of input sequences')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for creating sequences')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize the data')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Ratio of subjects for testing')
    parser.add_argument('--bidirectional', action='store_true',default=False,
                    help='Use bidirectional LSTM (BiLSTM)')
    
    parser.add_argument('--scaler_dir', type=str, default='./scalers_lstm')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--exclude_subject', type=str, default='Jovana')

    args = parser.parse_args()

    # Build loaders (shared)
    train_ds, val_ds, train_loader, val_loader, Fd, Jd, train_subs, test_subs = build_loaders_shared(
        data_dir=args.data_dir,
        seq_len=args.sequence_length,
        stride_train=args.stride,
        stride_val=args.sequence_length,   # disjoint for val
        batch_size=args.batch_size,
        normalize=args.normalize,
        scaler_dir=args.scaler_dir,
        test_ratio=args.test_ratio,
        exclude_subjects=[args.exclude_subject],
        num_workers=args.num_workers,
        scale_forces=True,
        scale_joints=False,  #(targets not normalized)
    )

    print(f"Training subjects: {train_subs}")
    print(f"Testing subjects:  {test_subs}")
    print(f"Training sequences: {len(train_ds)} | Validation sequences: {len(val_ds)}")

        
    # Create model
    model = ForcesTojointsLSTM(
                input_size=12,
                hidden_size=args.hidden_size,
                output_size=12,
                num_layers=args.num_layers,
                dropout=args.dropout,
                bidirectional=args.bidirectional
            )
    
    print(f"\nModel architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer and train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    trainer.train(num_epochs=args.num_epochs)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()