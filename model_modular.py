# -*- coding: utf-8 -*-
"""CNN-LSTM with contrastive gender encoder and dynamic memory (full model)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config as Config

from modules.contrastive_gender_encoder import ContrastiveGenderEncoder
from modules.dynamic_memory_network import DynamicMemoryEnhancementNetwork


class CNNLSTMFull(nn.Module):
    """CNN-LSTM with contrastive gender encoder and dynamic memory."""
    def __init__(self, input_dim=128, hidden_dim=96, num_layers=2, dropout=0.4,
                 output_dim=60, gender_embedding_dim=8):
        super(CNNLSTMFull, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.gender_embedding_dim = gender_embedding_dim
        self.use_contrastive_gender = True
        self.use_dynamic_memory = True
        self.use_gender = True
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        sequence_length = Config.SEQUENCE_LENGTH
        self.cnn_output_size = 128 * (input_dim // 8) * (sequence_length // 8)
        self.gender_embedding = nn.Embedding(2, gender_embedding_dim)
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.contrastive_gender_encoder = ContrastiveGenderEncoder(
            feature_dim=hidden_dim,
            prototype_dim=32,
            num_experts=2,
            contrastive_temperature=0.07,
            contrastive_margin=0.5
        )
        self.dynamic_memory_network = DynamicMemoryEnhancementNetwork(
            feature_dim=hidden_dim,
            flow_dim=output_dim,
            memory_size=1000,
            top_k=5,
            enhancement_weight=0.3
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, gender, compute_contrastive_loss=True):
        """Forward; returns dict with 'output', 'contrastive_loss', 'memory_info'."""
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(batch_size, -1, self.cnn_output_size)
        lstm_out, _ = self.lstm(x)
        lstm_last = lstm_out[:, -1, :]
        encoded_features, contrastive_loss, _ = \
            self.contrastive_gender_encoder(lstm_last, gender, compute_contrastive_loss)
        encoded_features_expanded = encoded_features.unsqueeze(1).expand(-1, lstm_out.size(1), -1)
        fused = lstm_out + encoded_features_expanded
        memory_info = {}
        if self.use_dynamic_memory:
            temp_output = self.fc(self.dropout(fused))
            temp_predicted_flows = temp_output.mean(dim=1)
            fused, memory_info = self.dynamic_memory_network(
                fused, 
                predicted_flows=temp_predicted_flows,
                update_memory=self.training
            )
        fused = self.dropout(fused)
        output = self.fc(fused)
        result = {
            'output': output,
            'contrastive_loss': contrastive_loss,
            'memory_info': memory_info
        }
        
        return result
    
    def get_module_info(self):
        """Return module info."""
        return {
            'use_contrastive_gender': True,
            'use_dynamic_memory': True,
            'modules_enabled': ['ContrastiveGenderEncoder', 'DynamicMemoryEnhancementNetwork']
        }


def create_model():
    """Build full CNN-LSTM model."""
    return CNNLSTMFull(
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        output_dim=Config.OUTPUT_DIM,
        gender_embedding_dim=8
    )


def get_model_info(model):
    """Return model parameter count and module info."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    module_info = model.get_module_info() if hasattr(model, 'get_module_info') else {}
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),
        'architecture': 'CNN-LSTM-Full',
        'input_dim': model.input_dim,
        'hidden_dim': model.hidden_dim,
        'num_layers': model.num_layers,
        'output_dim': model.output_dim,
        'gender_embedding_dim': model.gender_embedding_dim,
        **module_info
    }


if __name__ == "__main__":
    print("Testing full model...")
    model = create_model()
    info = get_model_info(model)
    print(f"Parameters: {info['total_parameters']:,}, Modules: {', '.join(info['modules_enabled'])}")
    x = torch.randn(2, 1, Config.INPUT_DIM, Config.SEQUENCE_LENGTH)
    g = torch.randint(0, 2, (2,))
    with torch.no_grad():
        out = model(x, g)['output']
    print(f"Output shape: {out.shape}, OK")

