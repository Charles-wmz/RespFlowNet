# -*- coding: utf-8 -*-
"""Dynamic memory enhancement network: retrieve similar history to augment features."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class DynamicMemoryBank(nn.Module):
    """Stores feature patterns and corresponding flow predictions for retrieval."""
    def __init__(self, 
                 feature_dim: int = 96,
                 flow_dim: int = 60,
                 memory_size: int = 1000,
                 similarity_threshold: float = 0.8,
                 confidence_threshold: float = 0.9):
        super(DynamicMemoryBank, self).__init__()
        
        self.feature_dim = feature_dim
        self.flow_dim = flow_dim
        self.memory_size = memory_size
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        self.register_buffer('memory_features', torch.zeros(memory_size, feature_dim))
        self.register_buffer('memory_flows', torch.zeros(memory_size, flow_dim))
        self.register_buffer('memory_confidences', torch.zeros(memory_size))
        self.register_buffer('memory_usage_count', torch.zeros(memory_size))
        self.register_buffer('memory_valid_mask', torch.zeros(memory_size, dtype=torch.bool))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        
    def compute_similarity(self, query_features: torch.Tensor, 
                          memory_features: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between query and memory features."""
        query_norm = F.normalize(query_features, p=2, dim=1)
        memory_norm = F.normalize(memory_features, p=2, dim=1)
        
        similarities = torch.mm(query_norm, memory_norm.t())
        return similarities
    
    def retrieve_memories(self, query_features: torch.Tensor, 
                         top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (retrieved_features, retrieved_flows, similarity_weights)."""
        batch_size = query_features.size(0)
        valid_memories = self.memory_features[self.memory_valid_mask]
        valid_flows = self.memory_flows[self.memory_valid_mask]
        
        if valid_memories.size(0) == 0:
            device = query_features.device
            retrieved_features = torch.zeros(batch_size, top_k, self.feature_dim, device=device)
            retrieved_flows = torch.zeros(batch_size, top_k, self.flow_dim, device=device)
            similarity_weights = torch.zeros(batch_size, top_k, device=device)
            return retrieved_features, retrieved_flows, similarity_weights
        similarities = self.compute_similarity(query_features, valid_memories)
        actual_k = min(top_k, valid_memories.size(0))
        top_similarities, top_indices = torch.topk(similarities, actual_k, dim=1)
        retrieved_features = valid_memories[top_indices]
        retrieved_flows = valid_flows[top_indices]
        similarity_weights = F.softmax(top_similarities / 0.1, dim=1)
        if actual_k < top_k:
            device = query_features.device
            pad_features = torch.zeros(batch_size, top_k - actual_k, self.feature_dim, device=device)
            pad_flows = torch.zeros(batch_size, top_k - actual_k, self.flow_dim, device=device)
            pad_weights = torch.zeros(batch_size, top_k - actual_k, device=device)
            
            retrieved_features = torch.cat([retrieved_features, pad_features], dim=1)
            retrieved_flows = torch.cat([retrieved_flows, pad_flows], dim=1)
            similarity_weights = torch.cat([similarity_weights, pad_weights], dim=1)
        
        return retrieved_features, retrieved_flows, similarity_weights
    
    def update_memory(self, new_features: torch.Tensor, 
                     new_flows: torch.Tensor,
                     confidences: torch.Tensor):
        """Update bank with high-confidence samples (training only)."""
        if not self.training:
            return
        batch_size = new_features.size(0)
        for i in range(batch_size):
            feature = new_features[i]
            flow = new_flows[i]
            confidence = confidences[i]
            if confidence < self.confidence_threshold:
                continue
            if self.memory_valid_mask.sum() > 0:
                similarities = self.compute_similarity(
                    feature.unsqueeze(0), 
                    self.memory_features[self.memory_valid_mask]
                )
                max_sim = similarities.max()
                
                if max_sim > self.similarity_threshold:
                    continue
            ptr = self.memory_ptr.item()
            self.memory_features[ptr] = feature.detach()
            self.memory_flows[ptr] = flow.detach()
            self.memory_confidences[ptr] = confidence.detach()
            self.memory_usage_count[ptr] = 0
            self.memory_valid_mask[ptr] = True
            self.memory_ptr[0] = (ptr + 1) % self.memory_size

    def get_memory_stats(self) -> Dict[str, float]:
        valid_count = self.memory_valid_mask.sum().item()
        avg_confidence = self.memory_confidences[self.memory_valid_mask].mean().item() if valid_count > 0 else 0.0
        avg_usage = self.memory_usage_count[self.memory_valid_mask].mean().item() if valid_count > 0 else 0.0
        
        return {
            'valid_memories': valid_count,
            'memory_utilization': valid_count / self.memory_size,
            'avg_confidence': avg_confidence,
            'avg_usage_count': avg_usage
        }


class DynamicMemoryEnhancementNetwork(nn.Module):
    """Retrieves similar history and enhances current features."""
    def __init__(self, 
                 feature_dim: int = 96,
                 flow_dim: int = 60,
                 memory_size: int = 1000,
                 top_k: int = 5,
                 enhancement_weight: float = 0.3):
        super(DynamicMemoryEnhancementNetwork, self).__init__()
        
        self.feature_dim = feature_dim
        self.flow_dim = flow_dim
        self.top_k = top_k
        self.enhancement_weight = enhancement_weight
        self.memory_bank = DynamicMemoryBank(
            feature_dim=feature_dim,
            flow_dim=flow_dim,
            memory_size=memory_size
        )
        self.memory_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        self.adaptive_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, lstm_features: torch.Tensor, 
                predicted_flows: Optional[torch.Tensor] = None,
                update_memory: bool = True) -> Tuple[torch.Tensor, Dict]:
        """Returns (enhanced_features, memory_info)."""
        batch_size, seq_len, feature_dim = lstm_features.shape
        query_features = lstm_features.mean(dim=1)
        retrieved_features, retrieved_flows, similarity_weights = \
            self.memory_bank.retrieve_memories(query_features, self.top_k)
        weighted_memory = (retrieved_features * similarity_weights.unsqueeze(-1)).sum(dim=1)
        combined_features = torch.cat([query_features, weighted_memory.view(batch_size, -1)], dim=1)
        fused_features = self.memory_fusion(combined_features)
        gate_weights = self.adaptive_gate(query_features)
        enhanced_query = query_features + self.enhancement_weight * gate_weights * fused_features
        enhanced_features = enhanced_query.unsqueeze(1).expand(-1, seq_len, -1)
        enhanced_features = lstm_features + self.enhancement_weight * (enhanced_features - lstm_features)
        memory_info = {'memory_stats': self.memory_bank.get_memory_stats()}
        if update_memory and predicted_flows is not None and self.training:
            flow_smoothness = -torch.var(predicted_flows, dim=1)
            confidences = torch.sigmoid(flow_smoothness)
            
            self.memory_bank.update_memory(query_features, predicted_flows, confidences)
            memory_info['updated_memories'] = True
        else:
            memory_info['updated_memories'] = False
        memory_info.update({
            'retrieved_count': (similarity_weights > 0.01).sum(dim=1).float().mean().item(),
            'avg_similarity': similarity_weights.max(dim=1)[0].mean().item(),
            'gate_activation': gate_weights.mean().item()
        })
        
        return enhanced_features, memory_info
    
    def reset_memory(self):
        """Clear memory bank."""
        self.memory_bank.memory_valid_mask.fill_(False)
        self.memory_bank.memory_ptr.fill_(0)
        
    def get_memory_info(self) -> Dict:
        return self.memory_bank.get_memory_stats()
