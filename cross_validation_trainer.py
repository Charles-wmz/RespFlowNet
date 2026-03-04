# -*- coding: utf-8 -*-
"""5-fold cross-validation trainer with gender input."""
import os
import time
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import pandas as pd
import random

from config import config as Config
from model_modular import create_model
from metrics import calculate_fvc_fev1

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CrossValidationTrainerWithGender:
    """5-fold cross-validation trainer with gender input."""
    
    def __init__(self, model, train_loader, val_loader, fold_idx, total_folds):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.fold_idx = fold_idx
        self.total_folds = total_folds
        
        print(f"Fold {fold_idx} - No normalization (using original scales: Mel=dB, Flow=L/s)")
        
        # Set random seed for this fold to ensure reproducibility
        # All folds should use the same seed for fair comparison
        fold_seed = Config.RANDOM_SEED
        set_seed(fold_seed)
        
        # Load true FVC and FEV1 labels
        self.load_true_labels()
        
        # Setup logging
        self.setup_logging()
        
        # Device configuration - auto select GPU if available
        if Config.DEVICE == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = torch.device('cpu')
                print("CUDA not available, using CPU")
        else:
            self.device = torch.device(Config.DEVICE)
        self.model.to(self.device)
        
        # Optimizer and loss function
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        # Learning rate scheduler setup
        if Config.LR_SCHEDULER_TYPE == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=Config.LR_SCHEDULER_FACTOR, 
                patience=Config.LR_SCHEDULER_PATIENCE,
                min_lr=Config.LR_SCHEDULER_MIN_LR
            )
        elif Config.LR_SCHEDULER_TYPE == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=Config.LR_SCHEDULER_PATIENCE, 
                gamma=Config.LR_SCHEDULER_FACTOR
            )
        elif Config.LR_SCHEDULER_TYPE == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=Config.EPOCHS,
                eta_min=Config.LR_SCHEDULER_MIN_LR
            )
        else:
            raise ValueError(f"Unknown scheduler type: {Config.LR_SCHEDULER_TYPE}")
        
        # Loss function selection
        loss_function = Config.LOSS_FUNCTION
        if loss_function == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_function == 'mae':
            self.criterion = nn.L1Loss()
        elif loss_function == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        
        # Logging
        self.log_dir = os.path.join(Config.OUTPUT_DIR, 'logs', f'fold_{fold_idx}')
        self.writer = SummaryWriter(self.log_dir)
        
        # Create output directories
        self.create_fold_dirs()
    
    def create_fold_dirs(self):
        """Create directories for current fold"""
        self.fold_output_dir = os.path.join(Config.OUTPUT_DIR, f'fold_{self.fold_idx}')
        self.fold_model_dir = os.path.join(self.fold_output_dir, 'models')
        self.fold_result_dir = os.path.join(self.fold_output_dir, 'results')
        
        os.makedirs(self.fold_model_dir, exist_ok=True)
        os.makedirs(self.fold_result_dir, exist_ok=True)
    
    def load_true_labels(self):
        """Load true FVC and FEV1 labels from CSV file"""
        try:
            # Load the label file (comma-separated)
            label_file = Config.LABEL_FILE
            self.true_labels_df = pd.read_csv(label_file)
            
            # Clean the data - remove spaces and convert to float
            self.true_labels_df['fvc'] = self.true_labels_df['fvc'].astype(str).str.strip().astype(float)
            self.true_labels_df['fev1'] = self.true_labels_df['fev1'].astype(str).str.strip().astype(float)
            
            # Create a mapping from file ID to true labels
            self.true_labels_map = {}
            for _, row in self.true_labels_df.iterrows():
                # Skip rows with NaN or invalid ID
                if pd.isna(row['id']) or not isinstance(row['id'], str):
                    continue
                    
                # Extract the first part of the ID (e.g., '0001' from '0001_2310714_20250320')
                file_id = str(row['id']).split('_')[0]
                self.true_labels_map[file_id] = {
                    'fvc': row['fvc'],
                    'fev1': row['fev1']
                }
            
            print(f"Loaded {len(self.true_labels_map)} true FVC/FEV1 labels")
            
        except Exception as e:
            print(f"Error loading true labels: {e}")
            self.true_labels_map = {}
    
    def get_true_labels(self, file_id):
        """Get true FVC and FEV1 for a given file ID"""
        # Extract the subject ID from the filename based on format
        if isinstance(file_id, str):
            if file_id.startswith(('c_', 'l_')):
                base_id = file_id.split('_')[1]  # Extract from c_0001_1 or l_0001_1
            else:
                base_id = file_id.split('_')[0]  # Extract from 0001_1
        else:
            file_id_str = str(file_id)
            if file_id_str.startswith(('c_', 'l_')):
                base_id = file_id_str.split('_')[1]  # Extract from c_0001_1 or l_0001_1
            else:
                base_id = file_id_str.split('_')[0]  # Extract from 0001_1
        
        return self.true_labels_map.get(base_id, {'fvc': None, 'fev1': None})
    
    def setup_logging(self):
        """Setup logging configuration"""
        # Create log directory if it doesn't exist
        log_dir = os.path.join(Config.OUTPUT_DIR, 'logs', f'fold_{self.fold_idx}')
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(f'cnn_lstm_cv_fold_{self.fold_idx}')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create file handler
        log_file = os.path.join(log_dir, f'fold_{self.fold_idx}_training.log')
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Log loss function
        loss_function = Config.LOSS_FUNCTION
        self.logger.info(f"Fold {self.fold_idx}/{self.total_folds} - Using loss function: {loss_function.upper()} + FEV1/FVC Ratio Loss")
    
    def calculate_ratio_loss(self, predicted_flow, time_batch, filenames):
        """
        Calculate FEV1/FVC ratio loss
        
        Args:
            predicted_flow: Predicted flow sequences (batch, seq_len)
            time_batch: Time sequences (batch, seq_len)
            filenames: List of filenames for getting true labels
        
        Returns:
            ratio_loss: MAE loss for FEV1/FVC ratio
        """
        batch_size = predicted_flow.shape[0]
        pred_ratios = []
        true_ratios = []
        
        # Convert to numpy for FVC/FEV1 calculation
        pred_flow_np = predicted_flow.detach().cpu().numpy()
        time_np = time_batch.cpu().numpy()
        
        for i in range(batch_size):
            # Get predicted FVC, FEV1, and PEF
            pred_fvc, pred_fev1, pred_pef = calculate_fvc_fev1(
                pred_flow_np[i], 
                time_np[i], 
                method="integration"
            )
            
            # Get true labels
            true_labels = self.get_true_labels(filenames[i])
            fvc_true = true_labels['fvc']
            fev1_true = true_labels['fev1']
            
            # Only calculate ratio loss if true labels are available
            if fvc_true is not None and fev1_true is not None and fvc_true > 0 and pred_fvc > 0:
                pred_ratio = pred_fev1 / pred_fvc
                true_ratio = fev1_true / fvc_true
                
                pred_ratios.append(pred_ratio)
                true_ratios.append(true_ratio)
        
        # Calculate ratio loss
        if len(pred_ratios) > 0:
            pred_ratios_tensor = torch.tensor(pred_ratios, dtype=torch.float32, device=predicted_flow.device)
            true_ratios_tensor = torch.tensor(true_ratios, dtype=torch.float32, device=predicted_flow.device)
            ratio_loss = nn.L1Loss()(pred_ratios_tensor, true_ratios_tensor)
        else:
            # If no valid samples in batch, return zero loss
            ratio_loss = torch.tensor(0.0, device=predicted_flow.device)
        
        return ratio_loss
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        total_flow_loss = 0.0
        total_ratio_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Fold {self.fold_idx} Epoch {self.current_epoch+1}/{Config.EPOCHS}')
        
        for batch_idx, (audio_batch, flow_batch, time_batch, gender_batch, filenames) in enumerate(pbar):
            # Move to device
            audio_batch = audio_batch.to(self.device)
            flow_batch = flow_batch.to(self.device)
            gender_batch = gender_batch.to(self.device)
            
            # Forward pass with gender input
            self.optimizer.zero_grad()
            predicted_flow = self.model(audio_batch, gender_batch)
            
            # Calculate loss - ensure dimensions match
            if predicted_flow.shape != flow_batch.shape:
                # If predicted_flow is (batch, 1, seq) and flow_batch is (batch, seq)
                if predicted_flow.dim() == 3 and flow_batch.dim() == 2:
                    # Remove the middle dimension if it's size 1
                    if predicted_flow.shape[1] == 1:
                        predicted_flow = predicted_flow.squeeze(1)
                    elif predicted_flow.shape[-1] == 1:
                        predicted_flow = predicted_flow.squeeze(-1)
                    else:
                        # If neither dimension is 1, squeeze the middle dimension
                        predicted_flow = predicted_flow.squeeze(1)
                # If predicted_flow is (batch, seq) and flow_batch is (batch, seq, 1)
                elif predicted_flow.dim() == 2 and flow_batch.dim() == 3:
                    flow_batch = flow_batch.squeeze(-1)
            
            # Calculate flow loss
            flow_loss = self.criterion(predicted_flow, flow_batch)
            
            # Calculate FEV1/FVC ratio loss
            ratio_loss = self.calculate_ratio_loss(predicted_flow, time_batch, filenames)
            
            # Combined loss: A * flow_loss + B * ratio_loss
            # Loss weights from config file
            loss = Config.FLOW_WEIGHT * flow_loss + Config.RATIO_WEIGHT * ratio_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total_flow_loss += flow_loss.item()
            total_ratio_loss += ratio_loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Flow': f'{flow_loss.item():.6f}',
                'Ratio': f'{ratio_loss.item():.6f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_flow_loss = total_flow_loss / num_batches
        avg_ratio_loss = total_ratio_loss / num_batches
        
        # Log individual losses
        self.logger.info(f'Epoch {self.current_epoch+1} - Train Loss: {avg_loss:.6f} (Flow: {avg_flow_loss:.6f}, Ratio: {avg_ratio_loss:.6f})')
        
        return avg_loss
    
    
    def validate_epoch(self):
        """Validate one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_flow_loss = 0.0
        total_ratio_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for audio_batch, flow_batch, time_batch, gender_batch, filenames in self.val_loader:
                # Move to device
                audio_batch = audio_batch.to(self.device)
                flow_batch = flow_batch.to(self.device)
                gender_batch = gender_batch.to(self.device)
                
                # Forward pass with gender input
                predicted_flow = self.model(audio_batch, gender_batch)
                
                # Calculate loss - ensure dimensions match
                if predicted_flow.shape != flow_batch.shape:
                    # If predicted_flow is (batch, 1, seq) and flow_batch is (batch, seq)
                    if predicted_flow.dim() == 3 and flow_batch.dim() == 2:
                        # Remove the middle dimension if it's size 1
                        if predicted_flow.shape[1] == 1:
                            predicted_flow = predicted_flow.squeeze(1)
                        elif predicted_flow.shape[-1] == 1:
                            predicted_flow = predicted_flow.squeeze(-1)
                        else:
                            # If neither dimension is 1, squeeze the middle dimension
                            predicted_flow = predicted_flow.squeeze(1)
                    # If predicted_flow is (batch, seq) and flow_batch is (batch, seq, 1)
                    elif predicted_flow.dim() == 2 and flow_batch.dim() == 3:
                        flow_batch = flow_batch.squeeze(-1)
                
                # Calculate flow loss
                flow_loss = self.criterion(predicted_flow, flow_batch)
                
                # Calculate FEV1/FVC ratio loss
                ratio_loss = self.calculate_ratio_loss(predicted_flow, time_batch, filenames)
                
                # Combined loss: A * flow_loss + B * ratio_loss
                # Loss weights from config file
                loss = Config.FLOW_WEIGHT * flow_loss + Config.RATIO_WEIGHT * ratio_loss
                
                # Statistics
                total_loss += loss.item()
                total_flow_loss += flow_loss.item()
                total_ratio_loss += ratio_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_flow_loss = total_flow_loss / num_batches
        avg_ratio_loss = total_ratio_loss / num_batches
        
        # Log individual losses
        self.logger.info(f'Epoch {self.current_epoch+1} - Val Loss: {avg_loss:.6f} (Flow: {avg_flow_loss:.6f}, Ratio: {avg_ratio_loss:.6f})')
        
        return avg_loss
    
    def evaluate_metrics(self, data_loader, dataset_name="Validation"):
        """Evaluate metrics"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_fvc_pred = []
        all_fvc_true = []
        all_fev1_pred = []
        all_fev1_true = []
        all_pef_pred = []
        all_pef_true = []
        
        # Store detailed sample information for CSV export
        sample_details = []
        
        with torch.no_grad():
            for audio_batch, flow_batch, time_batch, gender_batch, filenames in data_loader:
                # Move to device
                audio_batch = audio_batch.to(self.device)
                flow_batch = flow_batch.to(self.device)
                gender_batch = gender_batch.to(self.device)
                
                # Predict with gender input
                predicted_flow = self.model(audio_batch, gender_batch)
                
                # Collect predictions and targets - handle dimension mismatch
                pred_np = predicted_flow.cpu().numpy()
                target_np = flow_batch.cpu().numpy()
                
                # Ensure same dimensions before flattening
                if pred_np.shape != target_np.shape:
                    if pred_np.ndim == 3 and target_np.ndim == 2:
                        # Remove the middle dimension if it's size 1
                        if pred_np.shape[1] == 1:
                            pred_np = pred_np.squeeze(1)
                        elif pred_np.shape[-1] == 1:
                            pred_np = pred_np.squeeze(-1)
                        else:
                            # If neither dimension is 1, squeeze the middle dimension
                            pred_np = pred_np.squeeze(1)
                    elif pred_np.ndim == 2 and target_np.ndim == 3:
                        target_np = target_np.squeeze(-1)
                
                all_predictions.extend(pred_np.flatten())
                all_targets.extend(target_np.flatten())
                
                # Calculate FVC and FEV1 using true labels
                batch_size = flow_batch.size(0)
                for i in range(batch_size):
                    pred_flow_seq = predicted_flow[i].cpu().numpy().flatten()
                    flow_true_seq = flow_batch[i].cpu().numpy().flatten()
                    time_seq = time_batch[i].cpu().numpy().flatten()
                    filename = filenames[i]
                    
                    pred_flow_seq_denorm = pred_flow_seq
                    flow_true_seq_denorm = flow_true_seq
                    
                    # Get true FVC and FEV1 from labels
                    true_labels = self.get_true_labels(filename)
                    fvc_true = true_labels['fvc']
                    fev1_true = true_labels['fev1']
                    
                    fvc_pred, fev1_pred, pef_pred = calculate_fvc_fev1(pred_flow_seq_denorm, time_seq, method="integration")
                    
                    # Get true PEF (calculate from true flow sequence)
                    pef_true = np.max(flow_true_seq_denorm)
                    
                    # Calculate errors (absolute and relative)
                    fvc_abs_error = abs(fvc_pred - fvc_true) if fvc_true is not None else None
                    fev1_abs_error = abs(fev1_pred - fev1_true) if fev1_true is not None else None
                    pef_abs_error = abs(pef_pred - pef_true)
                    fvc_rel_error = (abs(fvc_pred - fvc_true) / max(fvc_true, 1e-8) * 100) if fvc_true is not None else None
                    fev1_rel_error = (abs(fev1_pred - fev1_true) / max(fev1_true, 1e-8) * 100) if fev1_true is not None else None
                    pef_rel_error = (abs(pef_pred - pef_true) / max(pef_true, 1e-8) * 100)
                    
                    flow_true_mean = np.mean(flow_true_seq_denorm)
                    flow_pred_mean = np.mean(pred_flow_seq_denorm)
                    flow_mae = np.mean(np.abs(flow_true_seq_denorm - pred_flow_seq_denorm))
                    flow_avg_error = np.mean(flow_true_seq_denorm - pred_flow_seq_denorm)
                    
                    # Store detailed information for each sample
                    # Extract subject ID based on filename format
                    if filename.startswith(('c_', 'l_')):
                        subject_id = filename.split('_')[1]  # Extract from c_0001_1 or l_0001_1
                    else:
                        subject_id = filename.split('_')[0]  # Extract from 0001_1
                    
                    sample_details.append({
                        'filename': filename,
                        'subject_id': subject_id,
                        'fvc_true': fvc_true,
                        'fvc_pred': fvc_pred,
                        'fev1_true': fev1_true,
                        'fev1_pred': fev1_pred,
                        'pef_true': pef_true,
                        'pef_pred': pef_pred,
                        'fvc_error': fvc_abs_error,
                        'fev1_error': fev1_abs_error,
                        'pef_error': pef_abs_error,
                        'fvc_error_pct': fvc_rel_error,
                        'fev1_error_pct': fev1_rel_error,
                        'pef_error_pct': pef_rel_error,
                        'fev1_fvc_ratio_true': fev1_true / fvc_true if fvc_true is not None and fvc_true != 0 else None,
                        'fev1_fvc_ratio_pred': fev1_pred / fvc_pred if fvc_pred != 0 else None,
                        'flow_true_mean': flow_true_mean,
                        'flow_pred_mean': flow_pred_mean,
                        'flow_mae': flow_mae,
                        'flow_avg_error': flow_avg_error
                    })
                    
                    # Only add to lists if we have true labels
                    if fvc_true is not None and fev1_true is not None:
                        all_fvc_pred.append(fvc_pred)
                        all_fvc_true.append(fvc_true)
                        all_fev1_pred.append(fev1_pred)
                        all_fev1_true.append(fev1_true)
                    
                    # Always add PEF (can be calculated from flow)
                    all_pef_pred.append(pef_pred)
                    all_pef_true.append(pef_true)
        
        # Calculate metrics
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Flow metrics
        flow_mae = mean_absolute_error(targets, predictions)
        flow_rmse = math.sqrt(mean_squared_error(targets, predictions))
        # R-squared for flow
        flow_ss_res = np.sum((targets - predictions) ** 2)
        flow_ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        flow_r_squared = 1 - (flow_ss_res / flow_ss_tot) if flow_ss_tot != 0 else 0
        
        # FVC and FEV1 metrics - only calculate if we have data
        if len(all_fvc_pred) > 0 and len(all_fvc_true) > 0:
            fvc_pred = np.array(all_fvc_pred)
            fvc_true = np.array(all_fvc_true)
            
            # FVC metrics
            fvc_mae = mean_absolute_error(fvc_true, fvc_pred)
            fvc_rmse = math.sqrt(mean_squared_error(fvc_true, fvc_pred))
            # MAPE with safety check for zero/small values
            fvc_mape = np.mean(np.abs((fvc_true - fvc_pred) / np.maximum(fvc_true, 1e-8))) * 100
            # R-squared for FVC
            fvc_ss_res = np.sum((fvc_true - fvc_pred) ** 2)
            fvc_ss_tot = np.sum((fvc_true - np.mean(fvc_true)) ** 2)
            fvc_r_squared = 1 - (fvc_ss_res / fvc_ss_tot) if fvc_ss_tot != 0 else 0
        else:
            # Set default values when no FVC data is available
            fvc_mae = fvc_rmse = fvc_mape = fvc_r_squared = 0.0
        
        if len(all_fev1_pred) > 0 and len(all_fev1_true) > 0:
            fev1_pred = np.array(all_fev1_pred)
            fev1_true = np.array(all_fev1_true)
            
            # FEV1 metrics
            fev1_mae = mean_absolute_error(fev1_true, fev1_pred)
            fev1_rmse = math.sqrt(mean_squared_error(fev1_true, fev1_pred))
            # MAPE with safety check for zero/small values
            fev1_mape = np.mean(np.abs((fev1_true - fev1_pred) / np.maximum(fev1_true, 1e-8))) * 100
            # R-squared for FEV1
            fev1_ss_res = np.sum((fev1_true - fev1_pred) ** 2)
            fev1_ss_tot = np.sum((fev1_true - np.mean(fev1_true)) ** 2)
            fev1_r_squared = 1 - (fev1_ss_res / fev1_ss_tot) if fev1_ss_tot != 0 else 0
        else:
            # Set default values when no FEV1 data is available
            fev1_mae = fev1_rmse = fev1_mape = fev1_r_squared = 0.0
        
        # PEF metrics - always available (calculated from flow)
        if len(all_pef_pred) > 0 and len(all_pef_true) > 0:
            pef_pred = np.array(all_pef_pred)
            pef_true = np.array(all_pef_true)
            
            # PEF metrics
            pef_mae = mean_absolute_error(pef_true, pef_pred)
            pef_rmse = math.sqrt(mean_squared_error(pef_true, pef_pred))
            # MAPE with safety check for zero/small values
            pef_mape = np.mean(np.abs((pef_true - pef_pred) / np.maximum(pef_true, 1e-8))) * 100
            # R-squared for PEF
            pef_ss_res = np.sum((pef_true - pef_pred) ** 2)
            pef_ss_tot = np.sum((pef_true - np.mean(pef_true)) ** 2)
            pef_r_squared = 1 - (pef_ss_res / pef_ss_tot) if pef_ss_tot != 0 else 0
        else:
            # Set default values when no PEF data is available
            pef_mae = pef_rmse = pef_mape = pef_r_squared = 0.0
        
        metrics = {
            'flow_mae': float(flow_mae),
            'flow_rmse': float(flow_rmse),
            'flow_r_squared': float(flow_r_squared),
            'fvc_mae': float(fvc_mae),
            'fvc_rmse': float(fvc_rmse),
            'fvc_mape': float(fvc_mape),
            'fvc_r_squared': float(fvc_r_squared),
            'fev1_mae': float(fev1_mae),
            'fev1_rmse': float(fev1_rmse),
            'fev1_mape': float(fev1_mape),
            'fev1_r_squared': float(fev1_r_squared),
            'pef_mae': float(pef_mae),
            'pef_rmse': float(pef_rmse),
            'pef_mape': float(pef_mape),
            'pef_r_squared': float(pef_r_squared),
            'sample_details': sample_details
        }
        
        self.logger.info(f"\nFold {self.fold_idx} - {dataset_name} Metrics:")
        self.logger.info(f"Flow - MAE: {flow_mae:.6f}, RMSE: {flow_rmse:.6f}, R²: {flow_r_squared:.6f}")
        self.logger.info(f"FVC - MAE: {fvc_mae:.6f}, RMSE: {fvc_rmse:.6f}, MAPE: {fvc_mape:.2f}%, R²: {fvc_r_squared:.6f}")
        self.logger.info(f"FEV1 - MAE: {fev1_mae:.6f}, RMSE: {fev1_rmse:.6f}, MAPE: {fev1_mape:.2f}%, R²: {fev1_r_squared:.6f}")
        self.logger.info(f"PEF - MAE: {pef_mae:.6f}, RMSE: {pef_rmse:.6f}, MAPE: {pef_mape:.2f}%, R²: {pef_r_squared:.6f}")
        
        return metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint - only save every 10 epochs and best model"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'fold_idx': self.fold_idx
        }
        
        # Only save checkpoint every 10 epochs (10, 20, 30, 40, 50, 60, ...)
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(self.fold_model_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f'  Saved checkpoint: {checkpoint_path}')
        
        # Always save best model
        if is_best:
            best_path = os.path.join(self.fold_model_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f'  Saved best model: {best_path}')
            self.logger.info(f"Fold {self.fold_idx} - New best model saved at epoch {epoch}")
    
    def plot_losses(self):
        """Plot loss curves"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Fold {self.fold_idx} - Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Training Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log Scale)')
        plt.title(f'Fold {self.fold_idx} - Training and Validation Loss (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.fold_result_dir, f'fold_{self.fold_idx}_loss_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_validation_samples_to_csv(self, sample_details):
        """Save validation sample details to CSV file"""
        import pandas as pd
        
        # Create DataFrame
        df = pd.DataFrame(sample_details)
        
        # Filter only samples with true labels
        df_with_labels = df[df['fvc_true'].notna() & df['fev1_true'].notna()].copy()
        
        # Sort by subject ID and filename
        df_with_labels = df_with_labels.sort_values(['subject_id', 'filename'])
        
        # Save to CSV
        csv_path = os.path.join(self.fold_result_dir, f'fold_{self.fold_idx}_validation_samples.csv')
        df_with_labels.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        self.logger.info(f"Fold {self.fold_idx} - Validation samples saved to: {csv_path}")
        self.logger.info(f"Fold {self.fold_idx} - Total validation samples: {len(df_with_labels)}")
        
        return csv_path
    
    def train(self):
        """Main training loop for current fold"""
        self.logger.info(f"Starting training for Fold {self.fold_idx}/{self.total_folds}...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(Config.EPOCHS):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate_epoch()
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            if Config.LR_SCHEDULER_TYPE == 'ReduceLROnPlateau':
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Log progress
            self.logger.info(f'Fold {self.fold_idx} Epoch {epoch+1}/{Config.EPOCHS}:')
            self.logger.info(f'  Train Loss: {train_loss:.6f}')
            self.logger.info(f'  Val Loss: {val_loss:.6f}')
            self.logger.info(f'  LR: {self.optimizer.param_groups[0]["lr"]:.8f}')
            
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint (every 10 epochs and best model)
            self.save_checkpoint(epoch, is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= Config.PATIENCE:
                self.logger.info(f"Fold {self.fold_idx} - Early stopping at epoch {epoch+1}")
                break
            
            self.logger.info('-' * 50)
        
        # Training completed
        training_time = time.time() - start_time
        self.logger.info(f"Fold {self.fold_idx} - Training completed in {training_time:.2f} seconds")
        
        # Load best model for final evaluation
        best_model_path = os.path.join(self.fold_model_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Fold {self.fold_idx} - Loaded best model for final evaluation")
        else:
            self.logger.warning(f"Fold {self.fold_idx} - Best model file not found, using current model")
        
        # Plot loss curves
        self.plot_losses()
        
        # Final evaluation on best model
        print(f"\nFold {self.fold_idx} - Final Evaluation (on best model):")
        train_metrics = self.evaluate_metrics(self.train_loader, "Training")
        val_metrics = self.evaluate_metrics(self.val_loader, "Validation")
        
        # Save validation sample details to CSV
        self.save_validation_samples_to_csv(val_metrics['sample_details'])
        
        # Save results
        results = {
            'fold': self.fold_idx,
            'training_time': training_time,
            'total_epochs': self.current_epoch + 1,
            'best_val_loss': self.best_val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': {
                'epochs': Config.EPOCHS,
                'batch_size': Config.BATCH_SIZE,
                'learning_rate': Config.LEARNING_RATE,
                'hidden_dim': Config.HIDDEN_DIM,
                'num_layers': Config.NUM_LAYERS,
                'dropout': Config.DROPOUT
            }
        }
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Convert results to JSON-serializable format
        results_serializable = convert_numpy_types(results)
        
        # Save as JSON
        with open(os.path.join(self.fold_result_dir, f'fold_{self.fold_idx}_results.json'), 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        # Save as TXT
        with open(os.path.join(self.fold_result_dir, f'fold_{self.fold_idx}_results.txt'), 'w') as f:
            f.write(f"CNN-LSTM Flow Prediction - Fold {self.fold_idx} Training Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Training Time: {training_time:.2f} seconds\n")
            f.write(f"Total Epochs: {self.current_epoch + 1}\n")
            f.write(f"Best Validation Loss: {self.best_val_loss:.6f}\n\n")
            
            f.write("Training Metrics:\n")
            for key, value in train_metrics.items():
                if key != 'sample_details':  # Skip sample_details as it's a list
                    f.write(f"  {key}: {value:.6f}\n")
            
            f.write("\nValidation Metrics:\n")
            for key, value in val_metrics.items():
                if key != 'sample_details':  # Skip sample_details as it's a list
                    f.write(f"  {key}: {value:.6f}\n")
        
        self.writer.close()
        return results

if __name__ == "__main__":
    # Test trainer
    from cross_validation_data_loader import create_cross_validation_data_loaders
    
    print("Testing cross validation trainer...")
    
    # Prepare data
    loader, folds = create_cross_validation_data_loaders()
    
    # Test first fold
    if folds:
        fold_info = folds[0]
        train_loader, val_loader = loader.create_fold_data_loaders(fold_info)
        
        # Create model
        model = create_model()
        
        # Create trainer
        trainer = CrossValidationTrainerWithGender(
            model, train_loader, val_loader, 
            fold_info['fold'], len(folds)
        )
        
        print("Cross validation trainer test completed successfully!")
