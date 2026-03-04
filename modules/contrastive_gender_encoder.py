# -*- coding: utf-8 -*-
"""Contrastive learning enhanced gender-adaptive encoder for respiratory flow prediction."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GenderPrototypeNetwork(nn.Module):
    """Learns male/female respiratory prototypes and projects features to prototype space."""
    def __init__(self, feature_dim, prototype_dim=32):
        super(GenderPrototypeNetwork, self).__init__()
        self.feature_dim = feature_dim
        self.prototype_dim = prototype_dim
        self.male_prototype = nn.Parameter(torch.randn(prototype_dim))
        self.female_prototype = nn.Parameter(torch.randn(prototype_dim))
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, prototype_dim * 2),
            nn.LayerNorm(prototype_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(prototype_dim * 2, prototype_dim),
            nn.LayerNorm(prototype_dim)
        )
        
    def forward(self, features):
        """Returns (projected_features, male_similarity, female_similarity)."""
        projected_features = self.feature_projection(features)
        projected_features_norm = F.normalize(projected_features, p=2, dim=1)
        male_prototype_norm = F.normalize(self.male_prototype, p=2, dim=0)
        female_prototype_norm = F.normalize(self.female_prototype, p=2, dim=0)
        male_similarity = torch.matmul(projected_features_norm, male_prototype_norm)
        female_similarity = torch.matmul(projected_features_norm, female_prototype_norm)  # (batch,)
        
        return projected_features, male_similarity, female_similarity


class ContrastiveLoss(nn.Module):
    """Pull same-gender samples together, push different-gender apart."""
    def __init__(self, temperature=0.07, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, projected_features, gender_labels):
        batch_size = projected_features.size(0)
        features_norm = F.normalize(projected_features, p=2, dim=1)
        similarity_matrix = torch.matmul(features_norm, features_norm.t()) / self.temperature
        gender_labels = gender_labels.unsqueeze(1)
        same_gender_mask = (gender_labels == gender_labels.t()).float()
        diff_gender_mask = (gender_labels != gender_labels.t()).float()
        mask_diag = torch.eye(batch_size, device=projected_features.device)
        same_gender_mask = same_gender_mask * (1 - mask_diag)
        positive_loss = -torch.log(torch.exp(similarity_matrix) / 
                                   (torch.exp(similarity_matrix).sum(dim=1, keepdim=True) + 1e-8) + 1e-8)
        positive_loss = (positive_loss * same_gender_mask).sum() / (same_gender_mask.sum() + 1e-8)
        negative_loss = torch.relu(similarity_matrix - self.margin)
        negative_loss = (negative_loss * diff_gender_mask).sum() / (diff_gender_mask.sum() + 1e-8)
        contrastive_loss = positive_loss + negative_loss
        
        return contrastive_loss


class DynamicGenderRouter(nn.Module):
    """Routes features by gender similarity using expert networks."""
    def __init__(self, feature_dim, num_experts=2):
        super(DynamicGenderRouter, self).__init__()
        self.feature_dim = feature_dim
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim)
            ) for _ in range(num_experts)
        ])
        self.router = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, features, male_similarity, female_similarity):
        """Returns (routed_features, expert_weights)."""
        batch_size = features.size(0)
        similarity_input = torch.stack([male_similarity, female_similarity], dim=1)
        expert_weights = self.router(similarity_input)
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(features))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        expert_weights_expanded = expert_weights.unsqueeze(2)
        routed_features = (expert_outputs * expert_weights_expanded).sum(dim=1) + features
        
        return routed_features, expert_weights


class ContrastiveGenderEncoder(nn.Module):
    """Combines prototype learning, contrastive loss, and dynamic routing."""
    def __init__(self, feature_dim, prototype_dim=32, num_experts=2, 
                 contrastive_temperature=0.07, contrastive_margin=0.5):
        super(ContrastiveGenderEncoder, self).__init__()
        self.feature_dim = feature_dim
        self.prototype_dim = prototype_dim
        self.prototype_network = GenderPrototypeNetwork(feature_dim, prototype_dim)
        self.contrastive_loss_fn = ContrastiveLoss(contrastive_temperature, contrastive_margin)
        self.dynamic_router = DynamicGenderRouter(feature_dim, num_experts)
        self.gender_embedding = nn.Embedding(2, 8)
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim + 8, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
        
    def forward(self, features, gender_labels, compute_contrastive_loss=True):
        """Returns (encoded_features, contrastive_loss, aux_info)."""
        batch_size = features.size(0)
        projected_features, male_similarity, female_similarity = self.prototype_network(features)
        contrastive_loss = None
        if compute_contrastive_loss and self.training:
            contrastive_loss = self.contrastive_loss_fn(projected_features, gender_labels)
        routed_features, expert_weights = self.dynamic_router(features, male_similarity, female_similarity)
        gender_emb = self.gender_embedding(gender_labels)
        combined = torch.cat([routed_features, gender_emb], dim=1)
        encoded_features = self.fusion_layer(combined)
        aux_info = {
            'male_similarity': male_similarity.detach(),
            'female_similarity': female_similarity.detach(),
            'expert_weights': expert_weights.detach(),
            'projected_features': projected_features.detach()
        }
        
        return encoded_features, contrastive_loss, aux_info
    
    def get_gender_statistics(self, aux_info, gender_labels):
        """Return per-gender similarity stats for analysis."""
        male_mask = (gender_labels == 0)
        female_mask = (gender_labels == 1)
        
        stats = {}
        
        if male_mask.sum() > 0:
            stats['male_to_male_prototype'] = aux_info['male_similarity'][male_mask].mean().item()
            stats['male_to_female_prototype'] = aux_info['female_similarity'][male_mask].mean().item()
        
        if female_mask.sum() > 0:
            stats['female_to_male_prototype'] = aux_info['male_similarity'][female_mask].mean().item()
            stats['female_to_female_prototype'] = aux_info['female_similarity'][female_mask].mean().item()
        
        return stats


if __name__ == "__main__":
    print("=" * 60)
    print("Testing ContrastiveGenderEncoder")
    print("=" * 60)
    batch_size = 8
    feature_dim = 96
    model = ContrastiveGenderEncoder(
        feature_dim=feature_dim,
        prototype_dim=32,
        num_experts=2,
        contrastive_temperature=0.07,
        contrastive_margin=0.5
    )
    features = torch.randn(batch_size, feature_dim)
    gender_labels = torch.randint(0, 2, (batch_size,))
    model.train()
    encoded_features, contrastive_loss, aux_info = model(features, gender_labels, compute_contrastive_loss=True)
    
    print(f"Input shape: {features.shape}, encoded: {encoded_features.shape}, contrastive loss: {contrastive_loss.item():.4f}")
    
    print("Prototype similarity stats:", model.get_gender_statistics(aux_info, gender_labels))
    print("Expert weights - male:", aux_info['expert_weights'][:, 0].mean().item(), 
          "female:", aux_info['expert_weights'][:, 1].mean().item())
    model.eval()
    with torch.no_grad():
        encoded_features_eval, contrastive_loss_eval, aux_info_eval = model(
            features, gender_labels, compute_contrastive_loss=False
        )
    
    print("Eval mode - encoded shape:", encoded_features_eval.shape)
    print("OK")
