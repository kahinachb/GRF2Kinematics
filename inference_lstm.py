"""
Inference and Visualization Script
Test the trained LSTM model and visualize predictions vs ground truth.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import argparse
from typing import Optional, Tuple
import pandas as pd
from train_lstm import ForcesTojointsLSTM, BiomechanicsSequenceDataset


class BiomechanicsInference:
    """Class for running inference and visualization with the trained model."""
    
    def __init__(self, 
                 model_path: str,
                 scaler_path: str = './scalers',
                 device: str = 'cuda'):
        """
        Initialize the inference class.
        
        Args:
            model_path: Path to trained model checkpoint
            scaler_path: Path to saved scalers
            device: Device to run inference on
        """
        self.device = device
        self.model = self.load_model(model_path)
        self.load_scalers(scaler_path)
        
        # Joint names for visualization
        self.joint_names = [
            'Left Hip Z', 'Left Hip X', 'Left Hip Y',
            'Left Knee Z', 'Left Ankle Z', 'Left Ankle X',
            'Right Hip Z', 'Right Hip X', 'Right Hip Y',
            'Right Knee Z', 'Right Ankle Z', 'Right Ankle X'
        ]
        
        self.force_names = [
            'Left Fx', 'Left Fy', 'Left Fz',
            'Left Mx', 'Left My', 'Left Mz',
            'Right Fx', 'Right Fy', 'Right Fz',
            'Right Mx', 'Right My', 'Right Mz'
            
        ]
    
    def load_model(self, model_path: str) -> nn.Module:
        """Load the trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model (ensure these match training parameters)
        # model = ForcesTojointsLSTM(
        #     input_size=12,
        #     hidden_size=128,  # Should match training
        #     output_size=12,
        #     num_layers=2,
        #     dropout=0
        # )
        
        # model.load_state_dict(checkpoint['model_state_dict'])

        ckpt = torch.load(model_path, map_location=self.device)
        hp = ckpt.get('model_hparams', {'input_size':12, 'hidden_size':128, 'output_size':12, 'num_layers':2, 'dropout':0.0})
        model = ForcesTojointsLSTM(**hp)
        model.load_state_dict(ckpt['model_state_dict'])

        model.to(self.device)
        model.eval()
        
        print(f"Loaded model from epoch {checkpoint['epoch']} with val loss: {checkpoint.get('val_loss', 'N/A')}")
        return model
    
    def load_scalers(self, scaler_path: str):
        """Load the scalers used during training."""
        scaler_dir = Path(scaler_path)
        
        with open(scaler_dir / 'force_scaler.pkl', 'rb') as f:
            self.force_scaler = pickle.load(f)
        # with open(scaler_dir / 'angle_scaler.pkl', 'rb') as f:
        #     self.angle_scaler = pickle.load(f)
    
    def predict_sequence(self, forces: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Predict joint joints from force sequence.
        
        Args:
            forces: Force sequence (sequence_length, 12)
            normalize: Whether to normalize input
            
        Returns:
            Predicted joints (sequence_length, 12)
        """
        # Normalize if needed
        if normalize:
            original_shape = forces.shape
            forces = self.force_scaler.transform(forces.reshape(-1, 12))
            forces = forces.reshape(original_shape)
        
        # Convert to tensor and add batch dimension
        forces_tensor = torch.from_numpy(forces.astype(np.float32)).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(forces_tensor)
        
        # Convert back to numpy
        predictions = predictions.squeeze(0).cpu().numpy()
        print(predictions.shape)
        
        # Denormalize
        # if normalize:
        #     original_shape = predictions.shape
        #     predictions = self.angle_scaler.inverse_transform(predictions.reshape(-1, 12))
        #     predictions = predictions.reshape(original_shape)
        
        return predictions

    def predict_last_frame(self, forces_window, normalize=True):
        """
        Predict only the last frame's joints from a window (T, 12).
        Returns: (12,)
        """
        
        forces = forces_window.astype(np.float32)

        if normalize:
            forces = self.force_scaler.transform(forces.reshape(-1, 12)).reshape(forces.shape)
        x = torch.from_numpy(forces).unsqueeze(0).to(self.device)   # (1, L, 12)
        with torch.no_grad():
            y_seq = self.model(x)            # (1, L, 12)
            y_last = y_seq[:, -1, :]         # (1, 12)
        y_last = y_last.squeeze(0).cpu().numpy()
        # if normalize:
        #     y_last = self.angle_scaler.inverse_transform(y_last.reshape(1, -1)).reshape(-1)
        return y_last

    
    def visualize_prediction(self,
                         forces: np.ndarray,
                         true_joints: np.ndarray,
                         pred_joints: np.ndarray,
                         title: str = "Force-to-Angle Prediction",
                         plot_in_degrees: bool = True):
        """
        Visualize prediction results.
        If plot_in_degrees=True, curves are converted to degrees for readability.
        """
        if plot_in_degrees:
            to_plot_true = true_joints * (180/np.pi)
            to_plot_pred = pred_joints * (180/np.pi)
            y_label = 'Angle (deg)'
        else:
            to_plot_true = true_joints
            to_plot_pred = pred_joints
            y_label = 'Angle (rad)'

        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)

        for i, (ax, joint_name) in enumerate(zip(axes.flat, self.joint_names)):
            time_steps = np.arange(len(to_plot_true))
            ax.plot(time_steps, to_plot_true[:, i], label='True', alpha=0.7)
            ax.plot(time_steps, to_plot_pred[:, i], label='Predicted', alpha=0.7)
            ax.set_title(joint_name)
            ax.set_xlabel('Time Step')
            ax.set_ylabel(y_label)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
    
    def calculate_metrics(self, true_joints: np.ndarray, pred_joints: np.ndarray) -> dict:
        """
        Calculate evaluation metrics.
        
        Args:
            true_joints: Ground truth joints
            pred_joints: Predicted joints
            
        Returns:
            Dictionary of metrics
        """

        # MSE per joint
        mse_per_joint = np.mean((true_joints - pred_joints) ** 2, axis=0)
        
        # RMSE per joint
        rmse_per_joint = np.sqrt(mse_per_joint)
        
        # MAE per joint
        mae_per_joint = np.mean(np.abs(true_joints - pred_joints), axis=0)
        
        # Overall metrics
        overall_mse = np.mean(mse_per_joint)
        overall_rmse = np.sqrt(overall_mse)
        overall_mae = np.mean(mae_per_joint)
        
        # Correlation per joint
        correlations = []
        for i in range(true_joints.shape[1]):
            corr = np.corrcoef(true_joints[:, i], pred_joints[:, i])[0, 1]
            correlations.append(corr)
        
        metrics = {
            'overall_mse': overall_mse,
            'overall_rmse': overall_rmse,
            'overall_mae': overall_mae,
            'mse_per_joint': mse_per_joint,
            'rmse_per_joint': rmse_per_joint,
            'mae_per_joint': mae_per_joint,
            'correlations': correlations
        }
        
        return metrics
    
    def print_metrics(self, metrics: dict, also_print_degrees: bool = True):
        """Print evaluation metrics (radians primary; optional degrees)."""
        
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)

        rmse_rad = metrics['overall_rmse']
        mae_rad  = metrics['overall_mae']
        mse_rad2 = metrics['overall_mse']

        print(f"\nOverall Metrics (radians):")
        print(f"  MSE:  {mse_rad2:.6f} rad^2")
        print(f"  RMSE: {rmse_rad:.6f} rad")
        print(f"  MAE:  {mae_rad:.6f} rad")

        if also_print_degrees:
            deg = 180/np.pi
            print(f"\nOverall Metrics (degrees):")
            print(f"  RMSE: {rmse_rad*deg:.3f} deg")
            print(f"  MAE:  {mae_rad*deg:.3f}  deg")

        print(f"\nPer-Joint Metrics:")
        print(f"{'Joint':<20} {'RMSE':>12} {'MAE':>12} {'Corr':>10}")
        print("-"*56)
        for i, joint_name in enumerate(self.joint_names):
            rmse = metrics['rmse_per_joint'][i]*deg
            mae  = metrics['mae_per_joint'][i]*deg
            corr = metrics['correlations'][i]
            if not np.isfinite(corr):  # guard for flat signals
                corr = 0.0
            print(f"{joint_name:<20} {rmse:>12.4f} {mae:>12.4f} {corr:>10.3f}")
        print("="*60)
    
    def test_on_file(self, force_file: str, angle_file: str):
        print(force_file)
        forces = np.load(force_file)
        joints = np.load(angle_file)

        print(f"Loaded data: forces shape {forces.shape}, joints shape {joints.shape}")

        sequence_length = 10  #used in training
        stride = 1            # 1 → predict for every frame (sliding window)
        predictions = []

        for i in range(sequence_length - 1, len(forces)):
            # extract the past 'sequence_length' frames up to current i
            window = forces[i - sequence_length + 1 : i + 1]
            y_last = self.predict_last_frame(window)
            predictions.append(y_last)

        predictions = np.vstack(predictions)              # (len(forces)-L+1, 12)
        true_joints = joints[sequence_length - 1:]        # align time axis
        # Compute metrics
        metrics = self.calculate_metrics(true_joints, predictions)
        self.print_metrics(metrics)

        # Visualize
        fig = self.visualize_prediction(
            forces[sequence_length - 1:], true_joints, predictions,
            title="Frame-by-frame last-step inference"
        )
        plt.show()

        return metrics
        
    def analyze_force_importance(self, sample_forces: np.ndarray):
        """
        Analyze which force components have the most influence on predictions.
        
        Args:
            sample_forces: Sample force sequence for analysis
        """
        base_prediction = self.predict_sequence(sample_forces)
        importance_scores = np.zeros(12)
        
        # Perturb each force component
        for i in range(12):
            perturbed_forces = sample_forces.copy()
            perturbed_forces[:, i] = 0  # Zero out this component
            
            perturbed_prediction = self.predict_sequence(perturbed_forces)
            
            # Calculate difference
            diff = np.mean(np.abs(base_prediction - perturbed_prediction))
            importance_scores[i] = diff
        
        # Normalize scores
        importance_scores = importance_scores / np.sum(importance_scores)
        
        # Plot importance
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red'] * 6 + ['blue'] * 6  # Right foot red, left foot blue
        bars = ax.bar(range(12), importance_scores, color=colors)
        ax.set_xticks(range(12))
        ax.set_xticklabels(self.force_names, rotation=45, ha='right')
        ax.set_ylabel('Relative Importance')
        ax.set_title('Force Component Importance for Joint Angle Prediction')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', label='Right Foot'),
                          Patch(facecolor='blue', label='Left Foot')]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.show()
        
        return importance_scores


def main():
    parser = argparse.ArgumentParser(description='Test and visualize the trained LSTM model')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing test data')
    parser.add_argument('--subject', type=str, required=True,
                        help='Subject to test on')
    parser.add_argument('--trial', type=str, required=True,
                        help='Trial to test on (e.g., trial107)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on')
    parser.add_argument('--analyze_importance', action='store_true',
                        help='Analyze force component importance')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = BiomechanicsInference(
        model_path=args.model_path,
        device=args.device
    )
    
    # Build file paths
    data_path = Path(args.data_dir) / args.subject / args.trial
    force_file = data_path / 'forces.npy'
    angle_file = data_path / 'joints.npy'
    
    if not force_file.exists() or not angle_file.exists():
        print(f"Error: Files not found for {args.subject}/{args.trial}")
        return
    
    print(f"\nTesting on {args.subject}/{args.trial}")
    print("-" * 50)
    
    # Test the model
    metrics = inference.test_on_file(force_file, angle_file)
    
    # Analyze force importance if requested
    if args.analyze_importance:
        print("\nAnalyzing force component importance...")
        forces = np.load(force_file)
        if len(forces) >= 50:
            sample_forces = forces[:50]  # Use first sequence
            importance = inference.analyze_force_importance(sample_forces)
            
            print("\nForce Component Importance:")
            for i, (name, score) in enumerate(zip(inference.force_names, importance)):
                print(f"  {name:<15}: {score:.3f}")


if __name__ == "__main__":
    main()