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


class BiomechanicsSequenceDataset(Dataset):
    """Dataset for loading biomechanical sequences by subject and task."""
    
    def __init__(self, 
                 data_dir: str,
                 sequence_length: int = 50,
                 stride: int = 10,
                 normalize: bool = True,
                 scaler_path: Optional[str] = None,
                 is_training: bool = True):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing processed NPY files
            sequence_length: Length of input sequences
            stride: Stride for creating overlapping sequences
            normalize: Whether to normalize the data
            scaler_path: Path to save/load scalers
            is_training: Whether this is training data (affects scaler fitting)
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        self.is_training = is_training
        self.scaler_path = scaler_path
        
        # Initialize scalers
        self.force_scaler = StandardScaler()
        # self.angle_scaler = StandardScaler()
        
        # Load all sequences
        self.sequences = []
        self.load_sequences()
        
        # Fit or load scalers
        if self.normalize:
            self.setup_scalers()
    
    def load_sequences(self):
        """Load all sequences organized by subject and trial."""
        print("Loading sequences...")
        
        # Iterate through subjects
        for subject_dir in sorted(self.data_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            # Iterate through trials
            for trial_dir in sorted(subject_dir.iterdir()):
                if not trial_dir.is_dir():
                    continue
                
                force_file = trial_dir / 'forces.npy'
                angle_file = trial_dir / 'joints.npy'
                
                if force_file.exists() and angle_file.exists():
                    forces = np.load(force_file)
                    joints = np.load(angle_file)
                    
                    # Create sequences for this trial (no overlap between trials)
                    trial_sequences = self.create_sequences(forces, joints)
                    
                    # Add metadata for tracking
                    for seq_forces, seq_joints in trial_sequences:
                        self.sequences.append({
                            'forces': seq_forces,
                            'joints': seq_joints,
                            'subject': subject_dir.name,
                            'trial': trial_dir.name
                        })
        
        print(f"Loaded {len(self.sequences)} sequences from {len(set(s['subject'] for s in self.sequences))} subjects")
    
    def create_sequences(self, forces: np.ndarray, joints: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create sequences from continuous data.
        
        Args:
            forces: Force data array (timesteps, features)
            joints: Angle data array (timesteps, features)
            
        Returns:
            List of (force_sequence, angle_sequence) tuples
        """
        sequences = []
        
        # Create overlapping sequences within the same trial
        for i in range(0, len(forces) - self.sequence_length + 1, self.stride):
            force_seq = forces[i:i + self.sequence_length]
            angle_seq = joints[i:i + self.sequence_length]
            sequences.append((force_seq, angle_seq))
        
        return sequences
    
    def setup_scalers(self):
        """Fit or load scalers for normalization."""
        scaler_dir = Path(self.scaler_path) if self.scaler_path else Path('./scalers')
        scaler_dir.mkdir(exist_ok=True)
        
        force_scaler_path = scaler_dir / 'force_scaler.pkl'
        # angle_scaler_path = scaler_dir / 'angle_scaler.pkl'
        
        if self.is_training:
            print("Fitting scalers on training data...")
            
            # Collect all data for fitting
            all_forces = []
            all_joints = []
            
            for seq in self.sequences:
                all_forces.append(seq['forces'].reshape(-1, seq['forces'].shape[-1]))
                all_joints.append(seq['joints'].reshape(-1, seq['joints'].shape[-1]))
            
            all_forces = np.vstack(all_forces)
            all_joints = np.vstack(all_joints)
            
            # Fit scalers
            self.force_scaler.fit(all_forces)
            # self.angle_scaler.fit(all_joints)
            
            # Save scalers
            with open(force_scaler_path, 'wb') as f:
                pickle.dump(self.force_scaler, f)
            # with open(angle_scaler_path, 'wb') as f:
            #     pickle.dump(self.angle_scaler, f)
            
            print("Scalers fitted and saved")
        else:
            # Load existing scalers
            if force_scaler_path.exists() :#and angle_scaler_path.exists():
                with open(force_scaler_path, 'rb') as f:
                    self.force_scaler = pickle.load(f)
                # with open(angle_scaler_path, 'rb') as f:
                #     self.angle_scaler = pickle.load(f)
                print("Scalers loaded from disk")
            else:
                print("Warning: No scalers found, normalization disabled")
                self.normalize = False
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Get a sequence pair."""
        seq_data = self.sequences[idx]
        forces = seq_data['forces'].astype(np.float32)
        joints = seq_data['joints'].astype(np.float32)
        
        # Normalize if enabled
        if self.normalize:
            # Reshape for scaling
            original_shape = forces.shape
            forces = self.force_scaler.transform(forces.reshape(-1, forces.shape[-1]))
            forces = forces.reshape(original_shape)
            
            # original_shape = joints.shape
            # joints = self.angle_scaler.transform(joints.reshape(-1, joints.shape[-1]))
            # joints = joints.reshape(original_shape)
        
        return torch.from_numpy(forces), torch.from_numpy(joints)


class ForcesTojointsLSTM(nn.Module):
    """LSTM model for predicting joint joints from force data."""
    
    def __init__(self, 
                 input_size: int = 12,  # 6D forces for each foot
                 hidden_size: int = 128,
                 output_size: int = 12,  # 6 joints for each leg
                 num_layers: int = 2,
                 dropout: float = 0.2):
        """
        Initialize the LSTM model.
        
        Args:
            input_size: Number of input features (force dimensions)
            hidden_size: Hidden state size
            output_size: Number of output features (joint joints)
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(ForcesTojointsLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Additional layers for better representation
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, output_size)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply fully connected layers to each time step
        # Reshape for FC layers: (batch_size * sequence_length, hidden_size)
        batch_size, seq_len, _ = lstm_out.shape
        lstm_out = lstm_out.reshape(-1, self.hidden_size)
        
        # Pass through FC layers
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Reshape back to (batch_size, sequence_length, output_size)
        out = out.reshape(batch_size, seq_len, self.output_size)
        
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
    
    def train(self, num_epochs: int, save_path: str = './checkpoints'):
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


def train_test_split_by_subject(data_dir: str, test_subjects: List[str] = None, test_ratio: float = 0.2):
    """
    Split data by subject for training and testing.
    
    Args:
        data_dir: Directory containing processed data
        test_subjects: List of subject names to use for testing
        test_ratio: Ratio of subjects to use for testing (if test_subjects not provided)
    
    Returns:
        train_subjects, test_subjects lists
    """
    data_path = Path(data_dir)
    all_subjects = [d.name for d in data_path.iterdir() if d.is_dir()]
    
    if test_subjects is None:
        # Randomly select test subjects
        num_test = max(1, int(len(all_subjects) * test_ratio))
        np.random.shuffle(all_subjects)
        test_subjects = all_subjects[:num_test]
        train_subjects = all_subjects[num_test:]
    else:
        train_subjects = [s for s in all_subjects if s not in test_subjects]
    
    return train_subjects, test_subjects


def main():
    parser = argparse.ArgumentParser(description='Train LSTM for force-to-angle prediction')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing processed NPY files')
    parser.add_argument('--sequence_length', type=int, default=10,
                        help='Length of input sequences')
    parser.add_argument('--stride', type=int, default=2,
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
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # Create train/test split by subject
    train_subjects, test_subjects = train_test_split_by_subject(args.data_dir, test_ratio=args.test_ratio)
    print(f"Training subjects: {train_subjects}")
    print(f"Testing subjects: {test_subjects}")
    
    # Create datasets
    train_dataset = BiomechanicsSequenceDataset(
        data_dir=args.data_dir, sequence_length=args.sequence_length,
        stride=args.stride, normalize=False, is_training=False
    )
    val_dataset = BiomechanicsSequenceDataset(
        data_dir=args.data_dir, sequence_length=args.sequence_length,
        stride=args.sequence_length, normalize=False, is_training=False
    )

    # Filter by subject first

    excluded_subject = "Jovana" #for validation

    train_subjects = [s for s in train_subjects if s != excluded_subject]
    test_subjects  = [s for s in test_subjects if s != excluded_subject]

    train_dataset.sequences = [s for s in train_dataset.sequences if s['subject'] in train_subjects]
    val_dataset.sequences   = [s for s in val_dataset.sequences   if s['subject'] in test_subjects]
    print(train_subjects)
    print("--")
    print(test_subjects)


    # fit on train only, then load for val
    train_dataset.normalize = False
    train_dataset.is_training = True
    train_dataset.setup_scalers()           # fits & saves using train only

    val_dataset.normalize = True
    val_dataset.is_training = False
    val_dataset.setup_scalers()             # loads the saved scalers

    print(f"Training sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = ForcesTojointsLSTM(
        input_size=12,  # 6D forces for each foot
        hidden_size=args.hidden_size,
        output_size=12,  # 6 joints for each leg
        num_layers=args.num_layers,
        dropout=args.dropout
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