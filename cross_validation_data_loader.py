# -*- coding: utf-8 -*-
"""5-fold cross-validation data loader with gender labels; uses all data, gender as model input."""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
import yaml
import pandas as pd
import random

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

def load_config():
    """Load configuration from YAML file"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class CrossValidationFlowDatasetWithGender(Dataset):
    """Dataset for cross validation flow data with gender labels"""
    
    def __init__(self, file_pairs, gender_labels, config, is_training=False):
        """
        Args:
            file_pairs: List of (mel_path, flow_path) tuples
            gender_labels: Dictionary mapping subject_id to gender (1=male, 0=female)
            config: Configuration dictionary
            is_training: Whether this is training dataset
        """
        self.file_pairs = file_pairs
        self.gender_labels = gender_labels
        self.config = config
        self.is_training = is_training
        
        print(f"Created dataset with {len(self.file_pairs)} samples (training: {is_training})")
        print(f"  Gender labels loaded for {len(gender_labels)} subjects")
        print(f"  No normalization - keeping original scales (Mel: dB, Flow: L/s)")
    
    def __len__(self):
        return len(self.file_pairs)
    
    def _extract_subject_id(self, filename):
        """Extract subject ID from filename"""
        if filename.startswith(('train_', 'val_', 'test_')):
            return filename.split('_')[1]
        elif filename.startswith(('c_', 'l_')):
            return filename.split('_')[1]
        else:
            return filename.split('_')[0]
    
    def __getitem__(self, idx):
        mel_path, flow_path = self.file_pairs[idx]
        
        # Extract filename and subject ID
        filename = os.path.basename(flow_path).replace('.csv', '')
        subject_id = self._extract_subject_id(filename)
        
        # Get gender label (default to 1 if not found, assuming male)
        gender = self.gender_labels.get(subject_id, 1)
        
        # Load preprocessed Mel spectrogram
        mel_spec = np.load(mel_path)
        
        # Load flow data from CSV
        flow_data = pd.read_csv(flow_path, header=None, names=['time', 'flow'])
        flow_sequence = flow_data['flow'].values
        time_sequence = flow_data['time'].values
        
        # Convert to tensors
        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)  # (1, 128, 60)
        flow_tensor = torch.FloatTensor(flow_sequence)
        time_tensor = torch.FloatTensor(time_sequence)
        gender_tensor = torch.LongTensor([gender])[0]  # Scalar tensor
        
        return mel_tensor, flow_tensor, time_tensor, gender_tensor, filename

class CrossValidationDataLoaderWithGender:
    """Data loader for 5-fold cross validation with gender labels"""
    
    def __init__(self, config):
        """Initialize data loader"""
        self.config = config
        self.mel_dir = "./data_aug/mel"
        self.csv_dir = "./data_aug/csv"
        
        # Load gender labels
        self.gender_labels = self._load_gender_labels()
        
        # Load all file pairs (no gender filtering, use all data)
        self.all_file_pairs = self._load_all_file_pairs()
        
        # Extract subject IDs for proper cross validation
        self.subject_ids = self._extract_subject_ids()
        
        print(f"Found {len(self.all_file_pairs)} total samples (ALL DATA)")
        print(f"Found {len(self.subject_ids)} unique subjects")
        
        # Print gender distribution
        gender_counts = {0: 0, 1: 0}
        for subject_id in self.subject_ids:
            gender = self.gender_labels.get(subject_id, 1)
            gender_counts[gender] += 1
        print(f"Gender distribution: Male={gender_counts[1]}, Female={gender_counts[0]}")
    
    def _load_gender_labels(self):
        """Load gender labels from Lung_fun_label.csv"""
        label_file = "./data/Lung_fun_label.csv"
        df = pd.read_csv(label_file)
        
        # Create a mapping from subject ID to gender
        gender_dict = {}
        for _, row in df.iterrows():
            subject_id = str(row['id']).split('_')[0]
            gender_dict[subject_id] = int(row['gender'])
        
        print(f"Loaded gender labels for {len(gender_dict)} subjects")
        return gender_dict
    
    def _load_all_file_pairs(self):
        """Load all mel-flow file pairs (no gender filtering)"""
        file_pairs = []
        
        if not os.path.exists(self.mel_dir):
            raise FileNotFoundError(f"Mel directory not found: {self.mel_dir}")
        
        if not os.path.exists(self.csv_dir):
            raise FileNotFoundError(f"CSV directory not found: {self.csv_dir}")
        
        # Get all mel files
        mel_files = sorted([f for f in os.listdir(self.mel_dir) if f.endswith('.npy')])
        
        for mel_file in mel_files:
            base_name = mel_file.replace('.npy', '')
            csv_file = f"{base_name}.csv"
            csv_path = os.path.join(self.csv_dir, csv_file)
            
            if os.path.exists(csv_path):
                mel_path = os.path.join(self.mel_dir, mel_file)
                file_pairs.append((mel_path, csv_path))
        
        return file_pairs
    
    def _extract_subject_ids(self):
        """Extract unique subject IDs from file pairs"""
        subject_ids = set()
        for wav_path, csv_path in self.all_file_pairs:
            filename = os.path.basename(csv_path).replace('.csv', '')
            
            if filename.startswith(('train_', 'val_', 'test_')):
                subject_id = filename.split('_')[1]
            elif filename.startswith(('c_', 'l_')):
                subject_id = filename.split('_')[1]
            else:
                subject_id = filename.split('_')[0]
            
            subject_ids.add(subject_id)
        
        return sorted(list(subject_ids))
    
    def create_cross_validation_folds(self, n_splits=5, random_state=42):
        """Create 5-fold cross validation splits using GroupKFold"""
        print(f"\nCreating {n_splits}-fold cross validation splits with GroupKFold...")
        print("Using ALL data (both male and female)")
        
        # Set random seed
        set_seed(random_state)
        
        # Group file pairs by subject ID
        subject_to_files = {}
        groups = []
        file_pairs_list = []
        
        for wav_path, csv_path in self.all_file_pairs:
            filename = os.path.basename(csv_path).replace('.csv', '')
            
            if filename.startswith(('train_', 'val_', 'test_')):
                subject_id = filename.split('_')[1]
            elif filename.startswith(('c_', 'l_')):
                subject_id = filename.split('_')[1]
            else:
                subject_id = filename.split('_')[0]
            
            if subject_id not in subject_to_files:
                subject_to_files[subject_id] = []
            subject_to_files[subject_id].append((wav_path, csv_path))
            
            file_pairs_list.append((wav_path, csv_path))
            groups.append(subject_id)
        
        # Create subject to group index mapping
        unique_subjects = sorted(list(set(groups)))
        subject_to_group_idx = {subject: idx for idx, subject in enumerate(unique_subjects)}
        groups_array = np.array([subject_to_group_idx[group] for group in groups])
        
        # Use GroupKFold for splitting
        group_kfold = GroupKFold(n_splits=n_splits)
        
        folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(group_kfold.split(file_pairs_list, groups=groups_array)):
            train_file_pairs = [file_pairs_list[i] for i in train_idx]
            val_file_pairs = [file_pairs_list[i] for i in val_idx]
            
            train_subjects = list(set([groups[i] for i in train_idx]))
            val_subjects = list(set([groups[i] for i in val_idx]))
            
            # Calculate gender distribution in this fold
            train_gender_dist = {0: 0, 1: 0}
            val_gender_dist = {0: 0, 1: 0}
            for subject in train_subjects:
                gender = self.gender_labels.get(subject, 0)
                train_gender_dist[gender] += 1
            for subject in val_subjects:
                gender = self.gender_labels.get(subject, 0)
                val_gender_dist[gender] += 1
            
            fold_info = {
                'fold': fold_idx + 1,
                'train_subjects': train_subjects,
                'val_subjects': val_subjects,
                'train_file_pairs': train_file_pairs,
                'val_file_pairs': val_file_pairs,
                'train_samples': len(train_file_pairs),
                'val_samples': len(val_file_pairs),
                'train_gender_dist': train_gender_dist,
                'val_gender_dist': val_gender_dist
            }
            
            folds.append(fold_info)
            
            print(f"\nFold {fold_idx + 1}:")
            print(f"  Train subjects: {len(train_subjects)} (Male={train_gender_dist[1]}, Female={train_gender_dist[0]})")
            print(f"  Val subjects: {len(val_subjects)} (Male={val_gender_dist[1]}, Female={val_gender_dist[0]})")
            print(f"  Train samples: {len(train_file_pairs)}")
            print(f"  Val samples: {len(val_file_pairs)}")
        
        return folds
    
    def create_fold_data_loaders(self, fold_info):
        """Create data loaders for a specific fold"""
        # Create datasets with gender labels
        train_dataset = CrossValidationFlowDatasetWithGender(
            fold_info['train_file_pairs'], 
            self.gender_labels,
            self.config, 
            is_training=True
        )
        val_dataset = CrossValidationFlowDatasetWithGender(
            fold_info['val_file_pairs'], 
            self.gender_labels,
            self.config, 
            is_training=False
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['preprocessing']['batch_size'], 
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['preprocessing']['batch_size'], 
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader

def create_cross_validation_data_loaders(n_splits=5, random_state=42):
    """
    Create cross validation data loaders with gender labels
    Uses ALL data (both male and female)
    """
    config = load_config()
    loader = CrossValidationDataLoaderWithGender(config)
    
    # Create cross validation folds
    folds = loader.create_cross_validation_folds(n_splits=n_splits, random_state=random_state)
    
    return loader, folds

if __name__ == "__main__":
    # Test the cross validation data loader
    print("Testing cross validation data loader with gender labels...")
    
    loader, folds = create_cross_validation_data_loaders()
    
    print(f"\nTotal folds created: {len(folds)}")
    
    # Test first fold
    if folds:
        fold_info = folds[0]
        train_loader, val_loader = loader.create_fold_data_loaders(fold_info)
        
        print(f"\nFold 1 Data Loaders:")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"No normalization - using original scales")
        
        # Test a batch
        for mel_batch, flow_batch, time_batch, gender_batch, filenames in train_loader:
            print(f"\nMel batch shape: {mel_batch.shape}")
            print(f"Flow batch shape: {flow_batch.shape}")
            print(f"Time batch shape: {time_batch.shape}")
            print(f"Gender batch shape: {gender_batch.shape}")
            print(f"Gender values: {gender_batch}")
            print(f"Mel value range: [{mel_batch.min():.2f}, {mel_batch.max():.2f}]")
            print(f"Flow value range: [{flow_batch.min():.2f}, {flow_batch.max():.2f}]")
            print(f"Filenames: {filenames[:3]}...")
            break

