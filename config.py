# -*- coding: utf-8 -*-
"""
Configuration loader for CNN-LSTM Flow Prediction Model
"""
import yaml
import os
import argparse

class Config:
    """Configuration class that loads from YAML file"""
    
    def __init__(self, config_path='config.yaml', override_args=None):
        """Load config from YAML; override_args e.g. {'training.epochs': 100}."""
        self.config_path = config_path
        self.load_config()
        if override_args:
            self.override_from_args(override_args)
    
    def load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Data paths
        self.DATA_ROOT = config['data']['root']
        self.WAV_DIR = config['data']['wav_dir']
        self.CSV_DIR = config['data']['csv_dir']
        self.LABEL_FILE = config['data']['label_file']
        
        # Output paths
        self.OUTPUT_DIR = config['output']['root']
        self.MODEL_DIR = config['output']['model_dir']
        self.LOG_DIR = config['output']['log_dir']
        self.RESULT_DIR = config['output']['result_dir']
        
        # Audio processing parameters
        self.SAMPLE_RATE = config['audio']['sample_rate']
        self.N_FFT = config['audio']['n_fft']
        self.HOP_LENGTH = config['audio']['hop_length']
        self.N_MELS = config['audio']['n_mels']
        self.MAX_AUDIO_LENGTH = config['audio']['max_audio_length']
        self.MEL_TIME_FRAMES = config['audio']['mel_time_frames']
        
        # Data preprocessing parameters
        self.SEQUENCE_LENGTH = config['preprocessing']['sequence_length']
        self.BATCH_SIZE = config['preprocessing']['batch_size']
        
        # Model parameters
        self.INPUT_DIM = config['model']['input_dim']
        self.HIDDEN_DIM = config['model']['hidden_dim']
        self.NUM_LAYERS = config['model']['num_layers']
        self.DROPOUT = config['model']['dropout']
        self.OUTPUT_DIM = config['model']['output_dim']
        
        # Training parameters
        self.EPOCHS = config['training']['epochs']
        self.LEARNING_RATE = float(config['training']['learning_rate'])
        self.WEIGHT_DECAY = float(config['training']['weight_decay'])
        self.PATIENCE = config['training']['patience']
        self.LOSS_FUNCTION = config['training']['loss_function']
        
        # Loss weights: A * flow_loss + B * ratio_loss
        self.FLOW_WEIGHT = float(config['training']['loss_weights']['flow_weight'])
        self.RATIO_WEIGHT = float(config['training']['loss_weights']['ratio_weight'])
        
        # Device configuration
        self.DEVICE = config['device']
        
        # Random seed settings
        self.RANDOM_SEED = config['random_seed']
        
        # FEV1 calculation parameters
        self.FEV1_METHOD = config['fev1']['method']
        self.FEV1_TIME = config['fev1']['fev1_time']
        
        # Data augmentation parameters
        self.DATA_AUG_ENABLED = config['data_augmentation']['enabled']
        self.TIME_STRETCH_RANGE = config['data_augmentation']['time_stretch_range']
        self.TIME_SHIFT_RANGE = config['data_augmentation']['time_shift_range']
        self.AUGMENTATION_PROBABILITY = config['data_augmentation']['augmentation_probability']
        
        # Learning rate scheduler parameters
        self.LR_SCHEDULER_TYPE = config['training']['lr_scheduler']['type']
        self.LR_SCHEDULER_FACTOR = config['training']['lr_scheduler']['factor']
        self.LR_SCHEDULER_PATIENCE = config['training']['lr_scheduler']['patience']
        self.LR_SCHEDULER_MIN_LR = config['training']['lr_scheduler']['min_lr']
    
    def get(self, key, default=None):
        """Get configuration value by key (supports nested keys like 'training.epochs')"""
        keys = key.split('.')
        value = self.__dict__
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def override_from_args(self, override_args):
        """Override config from dict of dotted keys (e.g. 'training.epochs') to values."""
        path_to_attr = {
            'data.root': 'DATA_ROOT',
            'data.wav_dir': 'WAV_DIR',
            'data.csv_dir': 'CSV_DIR',
            'data.label_file': 'LABEL_FILE',
            'output.root': 'OUTPUT_DIR',
            'output.model_dir': 'MODEL_DIR',
            'output.log_dir': 'LOG_DIR',
            'output.result_dir': 'RESULT_DIR',
            'audio.sample_rate': 'SAMPLE_RATE',
            'audio.n_fft': 'N_FFT',
            'audio.hop_length': 'HOP_LENGTH',
            'audio.n_mels': 'N_MELS',
            'audio.max_audio_length': 'MAX_AUDIO_LENGTH',
            'audio.mel_time_frames': 'MEL_TIME_FRAMES',
            'preprocessing.sequence_length': 'SEQUENCE_LENGTH',
            'preprocessing.batch_size': 'BATCH_SIZE',
            'model.input_dim': 'INPUT_DIM',
            'model.hidden_dim': 'HIDDEN_DIM',
            'model.num_layers': 'NUM_LAYERS',
            'model.dropout': 'DROPOUT',
            'model.output_dim': 'OUTPUT_DIM',
            'training.epochs': 'EPOCHS',
            'training.learning_rate': 'LEARNING_RATE',
            'training.weight_decay': 'WEIGHT_DECAY',
            'training.patience': 'PATIENCE',
            'training.loss_function': 'LOSS_FUNCTION',
            'training.loss_weights.flow_weight': 'FLOW_WEIGHT',
            'training.loss_weights.ratio_weight': 'RATIO_WEIGHT',
            'training.lr_scheduler.type': 'LR_SCHEDULER_TYPE',
            'training.lr_scheduler.factor': 'LR_SCHEDULER_FACTOR',
            'training.lr_scheduler.patience': 'LR_SCHEDULER_PATIENCE',
            'training.lr_scheduler.min_lr': 'LR_SCHEDULER_MIN_LR',
            'device': 'DEVICE',
            'random_seed': 'RANDOM_SEED',
            'fev1.method': 'FEV1_METHOD',
            'fev1.fev1_time': 'FEV1_TIME',
            'data_augmentation.enabled': 'DATA_AUG_ENABLED',
            'data_augmentation.time_stretch_range': 'TIME_STRETCH_RANGE',
            'data_augmentation.time_shift_range': 'TIME_SHIFT_RANGE',
            'data_augmentation.augmentation_probability': 'AUGMENTATION_PROBABILITY',
        }
        
        for key, value in override_args.items():
            if key in path_to_attr:
                attr_name = path_to_attr[key]
                original_value = getattr(self, attr_name, None)
                if original_value is not None:
                    if isinstance(original_value, bool):
                        value = bool(value) if not isinstance(value, bool) else value
                    elif isinstance(original_value, int):
                        value = int(value)
                    elif isinstance(original_value, float):
                        value = float(value)
                    elif isinstance(original_value, str):
                        value = str(value)
                setattr(self, attr_name, value)
                print(f"Config override: {key} = {value}")
            else:
                print(f"Warning: unknown config key '{key}'")
    
    def create_dirs(self):
        """Create necessary directories"""
        dirs = [self.OUTPUT_DIR, self.MODEL_DIR, self.LOG_DIR, self.RESULT_DIR]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)


def parse_config_overrides():
    """Parse --config key=value from argv. Returns dict of overrides."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', nargs='*', default=[],
                       help='Config overrides, e.g. training.epochs=100')
    args, unknown = parser.parse_known_args()
    
    override_dict = {}
    for config_arg in args.config:
        if '=' in config_arg:
            key, value = config_arg.split('=', 1)
            key = key.strip()
            value = value.strip()
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.lower() == 'none' or value.lower() == 'null':
                value = None
            else:
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass
            override_dict[key] = value
    
    return override_dict


def init_config(config_path='config.yaml', override_args=None):
    """Create global config; override_args None means parse from argv."""
    if override_args is None:
        override_args = parse_config_overrides()
    return Config(config_path, override_args)


config = Config()
