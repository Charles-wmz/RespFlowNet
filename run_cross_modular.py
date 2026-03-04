# -*- coding: utf-8 -*-
"""CNN-LSTM training: full model (contrastive gender encoder, dynamic memory, physics loss v2)."""
import os
import sys
import json
import time
import numpy as np
import torch
import random
from datetime import datetime
import pandas as pd
import logging

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config as Config, parse_config_overrides
from cross_validation_data_loader import create_cross_validation_data_loaders
from model_modular import create_model, get_model_info
from trainer_modular import ModularLossCalculator

from cross_validation_trainer import CrossValidationTrainerWithGender
import torch.nn as nn
from metrics import calculate_fvc_fev1, calculate_icc


class Tee:
    """Tee output to console and file."""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()


def setup_console_logging():
    """Redirect console output to both console and log file. Returns (log_file_path, original_stdout, log_file)."""
    log_dir = Config.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join(log_dir, f'console_output_{timestamp}.log')
    log_file = open(log_file_path, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)
    
    return log_file_path, original_stdout, log_file


def restore_console_logging(original_stdout, log_file):
    """Restore stdout and close log file."""
    sys.stdout = original_stdout
    if log_file:
        log_file.close()


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_cv_summary_results(all_fold_results, experiment_name):
    """Save cross-validation summary results."""
    cv_results_dir = os.path.join(Config.OUTPUT_DIR, 'cross_validation_results')
    os.makedirs(cv_results_dir, exist_ok=True)
    metrics_to_save = ['flow_mae', 'flow_rmse', 'flow_mape', 'flow_r_squared',
                      'fvc_mae', 'fvc_rmse', 'fvc_mape', 'fvc_r_squared', 'fvc_icc',
                      'fev1_mae', 'fev1_rmse', 'fev1_mape', 'fev1_r_squared', 'fev1_icc',
                      'fev1_fvc_mae', 'fev1_fvc_rmse', 'fev1_fvc_mape', 'fev1_fvc_icc',
                      'pef_mae', 'pef_rmse', 'pef_mape', 'pef_r_squared', 'pef_icc']
    
    train_metrics = {}
    for metric in metrics_to_save:
        if all_fold_results and metric in all_fold_results[0]['train_metrics']:
            train_metrics[metric] = [fold['train_metrics'][metric] for fold in all_fold_results]
    
    val_metrics = {}
    for metric in metrics_to_save:
        if all_fold_results and metric in all_fold_results[0]['val_metrics']:
            val_metrics[metric] = [fold['val_metrics'][metric] for fold in all_fold_results]
    
    cv_summary = {
        'train_metrics': {},
        'val_metrics': {},
        'training_time': {
            'total': sum([fold['training_time'] for fold in all_fold_results]),
            'mean': np.mean([fold['training_time'] for fold in all_fold_results]),
            'std': np.std([fold['training_time'] for fold in all_fold_results])
        },
        'epochs': {
            'mean': np.mean([fold['total_epochs'] for fold in all_fold_results]),
            'std': np.std([fold['total_epochs'] for fold in all_fold_results])
        }
    }
    
    for metric, values in train_metrics.items():
        cv_summary['train_metrics'][metric] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    for metric, values in val_metrics.items():
        cv_summary['val_metrics'][metric] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
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
    
    detailed_results = {
        'experiment_name': experiment_name,
        'cv_summary': convert_numpy_types(cv_summary),
        'individual_fold_results': convert_numpy_types(all_fold_results),
        'timestamp': datetime.now().isoformat(),
        'model_type': 'Modular CNN-LSTM with Gender Input',
        'config': {
            'epochs': Config.EPOCHS,
            'batch_size': Config.BATCH_SIZE,
            'learning_rate': Config.LEARNING_RATE,
            'hidden_dim': Config.HIDDEN_DIM,
            'num_layers': Config.NUM_LAYERS,
            'dropout': Config.DROPOUT,
            'random_seed': Config.RANDOM_SEED
        }
    }
    
    json_path = os.path.join(cv_results_dir, 'cross_validation_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    txt_path = os.path.join(cv_results_dir, 'cross_validation_results.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Modular CNN-LSTM - 5-Fold CV Results - Experiment: {experiment_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: Modular CNN-LSTM with Gender Input\n")
        f.write(f"Experiment: {experiment_name}\n\n")
        f.write("CV Summary (mean +/- std):\n")
        f.write("-" * 50 + "\n")
        f.write("\nValidation metrics:\n")
        for metric, stats in cv_summary['val_metrics'].items():
            metric_name = metric.upper().replace('_', ' ')
            if 'mape' in metric.lower():
                f.write(f"  {metric_name}: {stats['mean']:.2f} ± {stats['std']:.2f}%\n")
            else:
                f.write(f"  {metric_name}: {stats['mean']:.6f} ± {stats['std']:.6f}\n")
        
        f.write(f"\nTraining time: {cv_summary['training_time']['total']:.2f}s (total)\n")
        f.write(f"              {cv_summary['training_time']['mean']:.2f} +/- {cv_summary['training_time']['std']:.2f}s per fold\n")
        f.write(f"Epochs: {cv_summary['epochs']['mean']:.1f} +/- {cv_summary['epochs']['std']:.1f} per fold\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("Per-fold details:\n")
        f.write("=" * 80 + "\n")
        for i, fold_result in enumerate(all_fold_results):
            f.write(f"\nFold {i+1}:\n")
            f.write(f"  Training time: {fold_result['training_time']:.2f}s\n")
            f.write(f"  Epochs: {fold_result['total_epochs']}\n")
            f.write(f"  Best val loss: {fold_result['best_val_loss']:.6f}\n")
            if 'flow_mae' in fold_result['val_metrics']:
                flow_mape = fold_result['val_metrics'].get('flow_mape', 'N/A')
                flow_r2 = fold_result['val_metrics'].get('flow_r_squared', 'N/A')
                mape_str = f"{flow_mape:.2f}%" if isinstance(flow_mape, (int, float)) else str(flow_mape)
                r2_str = f"{flow_r2:.6f}" if isinstance(flow_r2, (int, float)) else str(flow_r2)
                f.write(f"  Val Flow MAE: {fold_result['val_metrics']['flow_mae']:.6f}, MAPE: {mape_str}, R²: {r2_str}\n")
            if 'fvc_mae' in fold_result['val_metrics']:
                fvc_icc = fold_result['val_metrics'].get('fvc_icc', 'N/A')
                icc_str = f"{fvc_icc:.6f}" if isinstance(fvc_icc, (int, float)) else str(fvc_icc)
                f.write(f"  Val FVC MAE: {fold_result['val_metrics']['fvc_mae']:.6f}, ICC: {icc_str}\n")
            if 'fev1_mae' in fold_result['val_metrics']:
                fev1_icc = fold_result['val_metrics'].get('fev1_icc', 'N/A')
                icc_str = f"{fev1_icc:.6f}" if isinstance(fev1_icc, (int, float)) else str(fev1_icc)
                f.write(f"  Val FEV1 MAE: {fold_result['val_metrics']['fev1_mae']:.6f}, ICC: {icc_str}\n")
            if 'fev1_fvc_mae' in fold_result['val_metrics']:
                fev1_fvc_icc = fold_result['val_metrics'].get('fev1_fvc_icc', 'N/A')
                icc_str = f"{fev1_fvc_icc:.6f}" if isinstance(fev1_fvc_icc, (int, float)) else str(fev1_fvc_icc)
                f.write(f"  Val FEV1/FVC MAE: {fold_result['val_metrics']['fev1_fvc_mae']:.6f}, ICC: {icc_str}\n")
            if 'pef_mae' in fold_result['val_metrics']:
                pef_icc = fold_result['val_metrics'].get('pef_icc', 'N/A')
                icc_str = f"{pef_icc:.6f}" if isinstance(pef_icc, (int, float)) else str(pef_icc)
                f.write(f"  Val PEF MAE: {fold_result['val_metrics']['pef_mae']:.6f}, ICC: {icc_str}\n")
    
    print("\nCollecting sample predictions...")
    all_samples_df = collect_all_sample_predictions()
    all_samples_csv_path = None
    stats_csv_path = None
    subject_stats_path = None
    
    if all_samples_df is not None and len(all_samples_df) > 0:
        all_samples_csv_path = os.path.join(cv_results_dir, 'all_samples_predictions.csv')
        all_samples_df.to_csv(all_samples_csv_path, index=False, encoding='utf-8-sig')
        
        print(f"  All sample predictions saved: {all_samples_csv_path}")
        print(f"  Total samples (all folds): {len(all_samples_df)}")
        stats_dict = {
            'subject_id': all_samples_df['subject_id'],
            'filename': all_samples_df['filename'],
            'fold': all_samples_df['fold'],
            'fvc_true': all_samples_df['fvc_true'],
            'fvc_pred': all_samples_df['fvc_pred'],
            'fvc_error': all_samples_df['fvc_pred'] - all_samples_df['fvc_true'],
            'fvc_abs_error': abs(all_samples_df['fvc_pred'] - all_samples_df['fvc_true']),
            'fvc_percent_error': abs((all_samples_df['fvc_pred'] - all_samples_df['fvc_true']) / all_samples_df['fvc_true'] * 100),
            'fev1_true': all_samples_df['fev1_true'],
            'fev1_pred': all_samples_df['fev1_pred'],
            'fev1_error': all_samples_df['fev1_pred'] - all_samples_df['fev1_true'],
            'fev1_abs_error': abs(all_samples_df['fev1_pred'] - all_samples_df['fev1_true']),
            'fev1_percent_error': abs((all_samples_df['fev1_pred'] - all_samples_df['fev1_true']) / all_samples_df['fev1_true'] * 100),
        }
        
        if 'gender' in all_samples_df.columns:
            stats_dict['gender'] = all_samples_df['gender']
        
        if 'flow_true_mean' in all_samples_df.columns:
            stats_dict['flow_true_mean'] = all_samples_df['flow_true_mean']
            stats_dict['flow_pred_mean'] = all_samples_df['flow_pred_mean']
            stats_dict['flow_mae'] = all_samples_df['flow_mae']
        
        stats_df = pd.DataFrame(stats_dict)
        
        stats_csv_path = os.path.join(cv_results_dir, 'sample_metrics_summary.csv')
        stats_df.to_csv(stats_csv_path, index=False, encoding='utf-8-sig')
        print(f"  Sample metrics summary saved: {stats_csv_path}")
        print(f"\nOverall sample statistics:")
        print(f"  FVC  - MAE: {stats_df['fvc_abs_error'].mean():.4f}, MAPE: {stats_df['fvc_percent_error'].mean():.2f}%")
        print(f"  FEV1 - MAE: {stats_df['fev1_abs_error'].mean():.4f}, MAPE: {stats_df['fev1_percent_error'].mean():.2f}%")
        
        if 'subject_id' in stats_df.columns:
            subject_stats = stats_df.groupby('subject_id').agg({
                'fvc_abs_error': 'mean',
                'fvc_percent_error': 'mean',
                'fev1_abs_error': 'mean',
                'fev1_percent_error': 'mean',
                'fold': 'first'
            }).reset_index()
            subject_stats.columns = ['subject_id', 'fvc_mae', 'fvc_mape', 'fev1_mae', 'fev1_mape', 'fold']
            subject_stats = subject_stats.sort_values(['fold', 'subject_id'])
            
            subject_stats_path = os.path.join(cv_results_dir, 'subject_level_statistics.csv')
            subject_stats.to_csv(subject_stats_path, index=False, encoding='utf-8-sig')
            print(f"  Subject-level stats saved: {subject_stats_path}")
            print(f"  Total subjects: {len(subject_stats)}")
    else:
        print("  Warning: No sample prediction files found; ensure validation samples are saved per fold.")
    print(f"\nCross-validation results saved:")
    print(f"  JSON: {json_path}")
    print(f"  TXT:  {txt_path}")
    if all_samples_df is not None and len(all_samples_df) > 0:
        print(f"  CSV - All samples: {all_samples_csv_path}")
        print(f"  CSV - Sample metrics: {stats_csv_path}")
        if subject_stats_path is not None:
            print(f"  CSV - Subject stats: {subject_stats_path}")


def collect_all_sample_predictions():
    """Collect sample predictions from all folds."""
    all_samples = []
    for fold_idx in range(1, 6):
        fold_csv_path = os.path.join(
            Config.OUTPUT_DIR,
            f'fold_{fold_idx}',
            'results',
            f'fold_{fold_idx}_validation_samples.csv'
        )
        
        if os.path.exists(fold_csv_path):
            try:
                df = pd.read_csv(fold_csv_path)
                df['fold'] = fold_idx
                all_samples.append(df)
            except Exception as e:
                print(f"  Warning: Could not read {fold_csv_path}: {e}")
    if all_samples:
        all_samples_df = pd.concat(all_samples, ignore_index=True)
        if 'subject_id' in all_samples_df.columns:
            all_samples_df = all_samples_df.sort_values(['fold', 'subject_id', 'filename'])
        else:
            all_samples_df = all_samples_df.sort_values(['fold', 'filename'])
        
        return all_samples_df
    else:
        return None


class ModularCrossValidationTrainer(CrossValidationTrainerWithGender):
    """Cross-validation trainer with physics loss v2 + contrastive loss."""
    def __init__(self, model, train_loader, val_loader, fold_idx, total_folds,
                 physics_loss_weights=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.fold_idx = fold_idx
        self.total_folds = total_folds
        self.load_true_labels()
        self.setup_logging()
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
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        if Config.LR_SCHEDULER_TYPE == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=Config.LR_SCHEDULER_FACTOR, 
                patience=Config.LR_SCHEDULER_PATIENCE,
                min_lr=Config.LR_SCHEDULER_MIN_LR
            )
        elif Config.LR_SCHEDULER_TYPE == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=Config.LR_SCHEDULER_PATIENCE, 
                gamma=Config.LR_SCHEDULER_FACTOR
            )
        elif Config.LR_SCHEDULER_TYPE == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=Config.EPOCHS,
                eta_min=Config.LR_SCHEDULER_MIN_LR
            )
        else:
            raise ValueError(f"Unknown scheduler type: {Config.LR_SCHEDULER_TYPE}")
        self.loss_calculator = ModularLossCalculator(
            physics_loss_weights=physics_loss_weights,
            contrastive_weight=0.1
        )
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.log_dir = os.path.join(Config.OUTPUT_DIR, 'logs', f'fold_{fold_idx}')
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
        except:
            self.writer = None
        self.create_fold_dirs()
        loss_info = self.loss_calculator.get_loss_info()
        print(f"\nFold {fold_idx} - Enabled losses: {', '.join(loss_info['loss_functions_enabled'])}")
    
    def train_epoch(self):
        """Train one epoch using modular loss calculator."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        from tqdm import tqdm
        pbar = tqdm(self.train_loader, desc=f'Fold {self.fold_idx} Epoch {self.current_epoch+1}/{Config.EPOCHS}')
        
        for batch_idx, (audio_batch, flow_batch, time_batch, gender_batch, filenames) in enumerate(pbar):
            audio_batch = audio_batch.to(self.device)
            flow_batch = flow_batch.to(self.device)
            time_batch = time_batch.to(self.device)
            gender_batch = gender_batch.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(audio_batch, gender_batch, compute_contrastive_loss=True)
            
            if isinstance(output, dict):
                predicted_flow = output['output']
                contrastive_loss = output.get('contrastive_loss', None)
                memory_info = output.get('memory_info', {})
            elif isinstance(output, tuple):
                predicted_flow, contrastive_loss = output
                memory_info = {}
            else:
                predicted_flow = output
                contrastive_loss = None
                memory_info = {}
            
            if predicted_flow.shape != flow_batch.shape:
                if predicted_flow.dim() == 3 and flow_batch.dim() == 2:
                    if predicted_flow.shape[1] == 1:
                        predicted_flow = predicted_flow.squeeze(1)
                    elif predicted_flow.shape[-1] == 1:
                        predicted_flow = predicted_flow.squeeze(-1)
                    else:
                        predicted_flow = predicted_flow.squeeze(1)
                elif predicted_flow.dim() == 2 and flow_batch.dim() == 3:
                    flow_batch = flow_batch.squeeze(-1)
            
            loss, loss_dict = self.loss_calculator.calculate_loss(
                predicted_flow, flow_batch, time_batch, 
                filenames, self.get_true_labels, contrastive_loss
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / num_batches
        self.logger.info(f'Epoch {self.current_epoch+1} - Train Loss: {avg_loss:.6f}')
        
        return avg_loss
    
    def validate_epoch(self):
        """Validate one epoch using modular loss calculator."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for audio_batch, flow_batch, time_batch, gender_batch, filenames in self.val_loader:
                audio_batch = audio_batch.to(self.device)
                flow_batch = flow_batch.to(self.device)
                time_batch = time_batch.to(self.device)
                gender_batch = gender_batch.to(self.device)
                output = self.model(audio_batch, gender_batch, compute_contrastive_loss=False)
                if isinstance(output, dict):
                    predicted_flow = output['output']
                elif isinstance(output, tuple):
                    predicted_flow, _ = output
                else:
                    predicted_flow = output
                if predicted_flow.shape != flow_batch.shape:
                    if predicted_flow.dim() == 3 and flow_batch.dim() == 2:
                        if predicted_flow.shape[1] == 1:
                            predicted_flow = predicted_flow.squeeze(1)
                        elif predicted_flow.shape[-1] == 1:
                            predicted_flow = predicted_flow.squeeze(-1)
                        else:
                            predicted_flow = predicted_flow.squeeze(1)
                    elif predicted_flow.dim() == 2 and flow_batch.dim() == 3:
                        flow_batch = flow_batch.squeeze(-1)
                
                loss, loss_dict = self.loss_calculator.calculate_loss(
                    predicted_flow, flow_batch, time_batch, 
                    filenames, self.get_true_labels, None
                )
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.logger.info(f'Epoch {self.current_epoch+1} - Val Loss: {avg_loss:.6f}')
        
        return avg_loss
    
    def load_true_labels(self):
        """Load true FVC and FEV1 labels from CSV file"""
        try:
            import pandas as pd
            label_file = Config.LABEL_FILE
            self.true_labels_df = pd.read_csv(label_file)
            self.true_labels_df['fvc'] = self.true_labels_df['fvc'].astype(str).str.strip().astype(float)
            self.true_labels_df['fev1'] = self.true_labels_df['fev1'].astype(str).str.strip().astype(float)
            self.true_labels_map = {}
            for _, row in self.true_labels_df.iterrows():
                if pd.isna(row['id']) or not isinstance(row['id'], str):
                    continue
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
        if isinstance(file_id, str):
            if file_id.startswith(('c_', 'l_')):
                base_id = file_id.split('_')[1]
            else:
                base_id = file_id.split('_')[0]
        else:
            file_id_str = str(file_id)
            if file_id_str.startswith(('c_', 'l_')):
                base_id = file_id_str.split('_')[1]
            else:
                base_id = file_id_str.split('_')[0]
        return self.true_labels_map.get(base_id, {'fvc': None, 'fev1': None})
    
    def setup_logging(self):
        """Setup logging configuration"""
        import logging
        log_dir = os.path.join(Config.OUTPUT_DIR, 'logs', f'fold_{self.fold_idx}')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger(f'cnn_lstm_cv_fold_{self.fold_idx}')
        self.logger.setLevel(logging.INFO)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        log_file = os.path.join(log_dir, f'fold_{self.fold_idx}_training.log')
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def create_fold_dirs(self):
        """Create directories for current fold"""
        self.fold_output_dir = os.path.join(Config.OUTPUT_DIR, f'fold_{self.fold_idx}')
        self.fold_model_dir = os.path.join(self.fold_output_dir, 'models')
        self.fold_result_dir = os.path.join(self.fold_output_dir, 'results')
        os.makedirs(self.fold_model_dir, exist_ok=True)
        os.makedirs(self.fold_result_dir, exist_ok=True)
    
    def evaluate_metrics(self, data_loader, dataset_name="Validation"):
        """Evaluate metrics; overridden to handle tuple/dict model output."""
        import numpy as np
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        sample_details = []
        
        with torch.no_grad():
            for audio_batch, flow_batch, time_batch, gender_batch, filenames in data_loader:
                audio_batch = audio_batch.to(self.device)
                flow_batch = flow_batch.to(self.device)
                gender_batch = gender_batch.to(self.device)
                output = self.model(audio_batch, gender_batch, compute_contrastive_loss=False)
                if isinstance(output, dict):
                    predicted_flow = output['output']
                elif isinstance(output, tuple):
                    predicted_flow, _ = output
                else:
                    predicted_flow = output
                pred_np = predicted_flow.cpu().numpy()
                target_np = flow_batch.cpu().numpy()
                
                if pred_np.shape != target_np.shape:
                    if pred_np.ndim == 3 and target_np.ndim == 2:
                        if pred_np.shape[1] == 1:
                            pred_np = pred_np.squeeze(1)
                        elif pred_np.shape[-1] == 1:
                            pred_np = pred_np.squeeze(-1)
                        else:
                            pred_np = pred_np.squeeze(1)
                    elif pred_np.ndim == 2 and target_np.ndim == 3:
                        target_np = target_np.squeeze(-1)
                
                all_predictions.extend(pred_np.flatten())
                all_targets.extend(target_np.flatten())
                
                batch_size = flow_batch.size(0)
                for i in range(batch_size):
                    pred_flow_seq = predicted_flow[i].cpu().numpy().flatten()
                    flow_true_seq = flow_batch[i].cpu().numpy().flatten()
                    time_seq = time_batch[i].cpu().numpy().flatten()
                    filename = filenames[i]
                    
                    pred_flow_seq_denorm = pred_flow_seq
                    flow_true_seq_denorm = flow_true_seq
                    true_labels = self.get_true_labels(filename)
                    fvc_true = true_labels['fvc']
                    fev1_true = true_labels['fev1']
                    
                    from metrics import calculate_fvc_fev1
                    fvc_pred, fev1_pred, pef_pred = calculate_fvc_fev1(
                        pred_flow_seq_denorm, time_seq, method="integration"
                    )
                    
                    pef_true = np.max(flow_true_seq_denorm)
                    if fvc_true is not None and fev1_true is not None:
                        subject_id = filename.split('_')[0] if '_' in filename else filename
                        
                        sample_details.append({
                            'filename': filename,
                            'subject_id': subject_id,
                            'fvc_true': float(fvc_true),
                            'fvc_pred': float(fvc_pred),
                            'fvc_error': float(abs(fvc_pred - fvc_true)),
                            'fev1_true': float(fev1_true),
                            'fev1_pred': float(fev1_pred),
                            'fev1_error': float(abs(fev1_pred - fev1_true)),
                            'pef_true': float(pef_true),
                            'pef_pred': float(pef_pred),
                            'pef_error': float(abs(pef_pred - pef_true))
                        })
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        flow_mae = mean_absolute_error(all_targets, all_predictions)
        flow_mse = mean_squared_error(all_targets, all_predictions)
        flow_rmse = np.sqrt(flow_mse)
        flow_r_squared = r2_score(all_targets, all_predictions)
        flow_mape = np.mean(np.abs((all_targets - all_predictions) / (np.abs(all_targets) + 1e-8))) * 100
        
        if len(sample_details) > 0:
            fvc_true_list = [s['fvc_true'] for s in sample_details]
            fvc_pred_list = [s['fvc_pred'] for s in sample_details]
            fev1_true_list = [s['fev1_true'] for s in sample_details]
            fev1_pred_list = [s['fev1_pred'] for s in sample_details]
            pef_true_list = [s['pef_true'] for s in sample_details]
            pef_pred_list = [s['pef_pred'] for s in sample_details]
            
            fvc_mae = mean_absolute_error(fvc_true_list, fvc_pred_list)
            fvc_rmse = np.sqrt(mean_squared_error(fvc_true_list, fvc_pred_list))
            fvc_r_squared = r2_score(fvc_true_list, fvc_pred_list)
            fvc_mape = np.mean(np.abs((np.array(fvc_true_list) - np.array(fvc_pred_list)) / 
                                     (np.array(fvc_true_list) + 1e-8))) * 100
            fvc_icc = calculate_icc(np.array(fvc_true_list), np.array(fvc_pred_list))
            
            fev1_mae = mean_absolute_error(fev1_true_list, fev1_pred_list)
            fev1_rmse = np.sqrt(mean_squared_error(fev1_true_list, fev1_pred_list))
            fev1_r_squared = r2_score(fev1_true_list, fev1_pred_list)
            fev1_mape = np.mean(np.abs((np.array(fev1_true_list) - np.array(fev1_pred_list)) / 
                                      (np.array(fev1_true_list) + 1e-8))) * 100
            fev1_icc = calculate_icc(np.array(fev1_true_list), np.array(fev1_pred_list))
            
            pef_mae = mean_absolute_error(pef_true_list, pef_pred_list)
            pef_rmse = np.sqrt(mean_squared_error(pef_true_list, pef_pred_list))
            pef_r_squared = r2_score(pef_true_list, pef_pred_list)
            pef_mape = np.mean(np.abs((np.array(pef_true_list) - np.array(pef_pred_list)) / 
                                     (np.array(pef_true_list) + 1e-8))) * 100
            pef_icc = calculate_icc(np.array(pef_true_list), np.array(pef_pred_list))
            
            fev1_fvc_true = np.array(fev1_true_list) / (np.array(fvc_true_list) + 1e-8)
            fev1_fvc_pred = np.array(fev1_pred_list) / (np.array(fvc_pred_list) + 1e-8)
            fev1_fvc_mae = mean_absolute_error(fev1_fvc_true, fev1_fvc_pred)
            fev1_fvc_rmse = np.sqrt(mean_squared_error(fev1_fvc_true, fev1_fvc_pred))
            fev1_fvc_mape = np.mean(np.abs((fev1_fvc_true - fev1_fvc_pred) / 
                                          (fev1_fvc_true + 1e-8))) * 100
            fev1_fvc_icc = calculate_icc(fev1_fvc_true, fev1_fvc_pred)
        else:
            fvc_mae = fvc_rmse = fvc_r_squared = fvc_mape = fvc_icc = 0.0
            fev1_mae = fev1_rmse = fev1_r_squared = fev1_mape = fev1_icc = 0.0
            pef_mae = pef_rmse = pef_r_squared = pef_mape = pef_icc = 0.0
            fev1_fvc_mae = fev1_fvc_rmse = fev1_fvc_mape = fev1_fvc_icc = 0.0
        
        metrics = {
            'flow_mae': flow_mae,
            'flow_rmse': flow_rmse,
            'flow_mape': flow_mape,
            'flow_r_squared': flow_r_squared,
            'fvc_mae': fvc_mae,
            'fvc_rmse': fvc_rmse,
            'fvc_mape': fvc_mape,
            'fvc_r_squared': fvc_r_squared,
            'fvc_icc': fvc_icc,
            'fev1_mae': fev1_mae,
            'fev1_rmse': fev1_rmse,
            'fev1_mape': fev1_mape,
            'fev1_r_squared': fev1_r_squared,
            'fev1_icc': fev1_icc,
            'fev1_fvc_mae': fev1_fvc_mae,
            'fev1_fvc_rmse': fev1_fvc_rmse,
            'fev1_fvc_mape': fev1_fvc_mape,
            'fev1_fvc_icc': fev1_fvc_icc,
            'pef_mae': pef_mae,
            'pef_rmse': pef_rmse,
            'pef_mape': pef_mape,
            'pef_r_squared': pef_r_squared,
            'pef_icc': pef_icc,
            'sample_details': sample_details
        }
        
        print(f"\n{dataset_name} Metrics:")
        print(f"  Flow MAE: {flow_mae:.6f}, RMSE: {flow_rmse:.6f}, MAPE: {flow_mape:.2f}%, R²: {flow_r_squared:.6f}")
        print(f"  FVC MAE: {fvc_mae:.6f}, RMSE: {fvc_rmse:.6f}, MAPE: {fvc_mape:.2f}%, R²: {fvc_r_squared:.6f}, ICC: {fvc_icc:.6f}")
        print(f"  FEV1 MAE: {fev1_mae:.6f}, RMSE: {fev1_rmse:.6f}, MAPE: {fev1_mape:.2f}%, R²: {fev1_r_squared:.6f}, ICC: {fev1_icc:.6f}")
        print(f"  FEV1/FVC MAE: {fev1_fvc_mae:.6f}, RMSE: {fev1_fvc_rmse:.6f}, MAPE: {fev1_fvc_mape:.2f}%, ICC: {fev1_fvc_icc:.6f}")
        print(f"  PEF MAE: {pef_mae:.6f}, RMSE: {pef_rmse:.6f}, MAPE: {pef_mape:.2f}%, R²: {pef_r_squared:.6f}, ICC: {pef_icc:.6f}")
        
        return metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint; only best model is saved."""
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
        if is_best:
            best_path = os.path.join(self.fold_model_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f'  Saved best model: {best_path}')
            self.logger.info(f"Fold {self.fold_idx} - New best model saved at epoch {epoch+1}")
    
    def plot_losses(self):
        """Plot loss curves (delegates to base trainer)."""
        return CrossValidationTrainerWithGender.plot_losses(self)
    
    def save_validation_samples_to_csv(self, sample_details):
        """Save validation samples (delegates to base trainer)."""
        return CrossValidationTrainerWithGender.save_validation_samples_to_csv(self, sample_details)
    
    def train(self):
        """Main training loop using modular loss."""
        import time
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
            if self.writer:
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
            
            # Save checkpoint
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
        
        # Plot loss curves
        self.plot_losses()
        
        # Final evaluation
        print(f"\nFold {self.fold_idx} - Final Evaluation (on best model):")
        train_metrics = self.evaluate_metrics(self.train_loader, "Training")
        val_metrics = self.evaluate_metrics(self.val_loader, "Validation")
        
        # Save validation sample details
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
        
        # Convert numpy types to Python types
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
        
        results_serializable = convert_numpy_types(results)
        
        # Save as JSON
        with open(os.path.join(self.fold_result_dir, f'fold_{self.fold_idx}_results.json'), 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        if self.writer:
            self.writer.close()
        
        return results


def main(experiment_name="run"):
    """Run 5-fold cross-validation with full CNN-LSTM model."""
    original_output_dir = Config.OUTPUT_DIR
    experiment_output_dir = os.path.join(original_output_dir, experiment_name)
    Config.OUTPUT_DIR = experiment_output_dir
    Config.MODEL_DIR = os.path.join(experiment_output_dir, 'models')
    Config.LOG_DIR = os.path.join(experiment_output_dir, 'logs')
    Config.RESULT_DIR = os.path.join(experiment_output_dir, 'results')
    Config.create_dirs()
    log_file_path, original_stdout, log_file = setup_console_logging()
    try:
        print("=" * 80)
        print(f"CNN-LSTM (full model) - Run: {experiment_name}")
        print("=" * 80)
        print(f"Device: {Config.DEVICE}, Seed: {Config.RANDOM_SEED}")
        print(f"Data: ./data_aug/, Output: {Config.OUTPUT_DIR}/")
        print("Model: contrastive gender encoder + dynamic memory + physics loss v2")
        print("=" * 80)
        total_start_time = time.time()
        set_seed(Config.RANDOM_SEED)
        print(f"\n1. Preparing cross-validation data...")
        loader, folds = create_cross_validation_data_loaders(
            n_splits=5, 
            random_state=Config.RANDOM_SEED
        )
        
        print(f"   Total folds: {len(folds)}")
        for i, fold in enumerate(folds):
            print(f"   Fold {i+1}: {fold['train_samples']} train, {fold['val_samples']} val samples")
        all_fold_results = []
        for fold_idx, fold_info in enumerate(folds):
            print(f"\n{'='*60}")
            print(f"Training Fold {fold_idx + 1}/{len(folds)}")
            print(f"{'='*60}")
            train_loader, val_loader = loader.create_fold_data_loaders(fold_info)
            print(f"   Train batches: {len(train_loader)}")
            print(f"   Val batches: {len(val_loader)}")
            print(f"\n2. Creating model...")
            fold_seed = Config.RANDOM_SEED
            set_seed(fold_seed)
            model = create_model()
            model_info = get_model_info(model)
            print(f"   Model parameters: {model_info['total_parameters']:,}")
            print(f"\n3. Training...")
            physics_weights = {
                'flow': 2.0,
                'fvc_integral': 1.5,
                'fev1_integral': 1.5,
                'smoothness': 0.3
            }
            trainer = ModularCrossValidationTrainer(
                model, train_loader, val_loader,
                fold_idx + 1, len(folds),
                physics_loss_weights=physics_weights
            )
            fold_results = trainer.train()
            all_fold_results.append(fold_results)
            loss_curve_path = os.path.join(
                Config.OUTPUT_DIR, 
                f'fold_{fold_idx + 1}', 
                'results', 
                f'fold_{fold_idx + 1}_loss_curves.png'
            )
            
            print(f"\nFold {fold_idx + 1} done.")
            print(f"   Best val loss: {fold_results['best_val_loss']:.6f}")
            print(f"   Epochs: {fold_results['total_epochs']}")
            print(f"   Training time: {fold_results['training_time']:.2f}s")
            print(f"   Loss curve: {loss_curve_path}")
        print(f"\n{'='*80}")
        print(f"5-Fold CV Summary - Experiment: {experiment_name}")
        print(f"{'='*80}")
        metrics_to_average = ['flow_mae', 'flow_rmse', 'flow_mape', 'flow_r_squared',
                             'fvc_mae', 'fvc_rmse', 'fvc_mape', 'fvc_r_squared', 'fvc_icc',
                             'fev1_mae', 'fev1_rmse', 'fev1_mape', 'fev1_r_squared', 'fev1_icc',
                             'fev1_fvc_mae', 'fev1_fvc_rmse', 'fev1_fvc_mape', 'fev1_fvc_icc',
                             'pef_mae', 'pef_rmse', 'pef_mape', 'pef_r_squared', 'pef_icc']
        
        print(f"\nValidation metrics (mean +/- std):")
        for metric in metrics_to_average:
            if all_fold_results and metric in all_fold_results[0]['val_metrics']:
                values = [fold['val_metrics'][metric] for fold in all_fold_results]
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"  {metric.upper().replace('_', ' ')}: {mean_val:.6f} ± {std_val:.6f}")
        
        save_cv_summary_results(all_fold_results, experiment_name)
        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time
        total_minutes = int(total_elapsed_time // 60)
        total_seconds = int(total_elapsed_time % 60)
        
        print(f"\n{'='*80}")
        print("Training/validation loss curves:")
        print(f"{'='*80}")
        for fold_idx in range(len(all_fold_results)):
            loss_curve_path = os.path.join(
                Config.OUTPUT_DIR, 
                f'fold_{fold_idx + 1}', 
                'results', 
                f'fold_{fold_idx + 1}_loss_curves.png'
            )
            if os.path.exists(loss_curve_path):
                print(f"  Fold {fold_idx + 1}: {loss_curve_path}")
            else:
                print(f"  Fold {fold_idx + 1}: loss curve not found")
        print(f"\n{'='*80}")
        print(f"Training complete. Experiment: {experiment_name}")
        print(f"Results: {Config.OUTPUT_DIR}")
        print(f"Total time: {total_minutes}m {total_seconds}s")
        print(f"Console log: {log_file_path}")
        print(f"{'='*80}")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        restore_console_logging(original_stdout, log_file)


if __name__ == "__main__":
    import argparse
    override_args = parse_config_overrides()
    if override_args:
        print("=" * 60)
        print("Applying config overrides...")
        print("=" * 60)
        Config.override_from_args(override_args)
        print("=" * 60)
        print()
    parser = argparse.ArgumentParser(description='CNN-LSTM 5-fold CV training (full model)')
    parser.add_argument('--config', nargs='*', default=[], help='Config overrides (key=value)')
    parser.add_argument('--experiment', type=str, default='run', help='Run name / output subdir')
    args = parser.parse_args()
    main(experiment_name=args.experiment)

