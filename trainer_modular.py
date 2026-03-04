# -*- coding: utf-8 -*-
"""Loss: flow + physics-informed v2 (FVC/FEV1 integral + smoothness) + contrastive."""
import torch
import torch.nn as nn
from modules.physics_loss_v2 import PhysicsInformedLossV2
from config import config as Config
from metrics import calculate_fvc_fev1


class ModularLossCalculator:
    """Flow loss + physics loss v2 + contrastive loss."""
    def __init__(self, physics_loss_weights=None, contrastive_weight=0.1):
        self.contrastive_weight = contrastive_weight
        if physics_loss_weights is None:
            physics_loss_weights = {
                'flow': 2.0,
                'fvc_integral': 1.5,
                'fev1_integral': 1.5,
                'smoothness': 0.3
            }
        self.physics_loss = PhysicsInformedLossV2(
            flow_weight=physics_loss_weights['flow'],
            fvc_integral_weight=physics_loss_weights.get('fvc_integral', 1.5),
            fev1_integral_weight=physics_loss_weights.get('fev1_integral', 1.5),
            smoothness_weight=physics_loss_weights.get('smoothness', 0.3)
        )
        loss_function = Config.LOSS_FUNCTION
        if loss_function == 'mse':
            self.base_criterion = nn.MSELoss()
        elif loss_function == 'mae':
            self.base_criterion = nn.L1Loss()
        elif loss_function == 'smooth_l1':
            self.base_criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
        self.flow_weight = Config.FLOW_WEIGHT
        self.ratio_weight = Config.RATIO_WEIGHT
    
    def calculate_loss(self, predicted_flow, flow_batch, time_batch, filenames, 
                      get_true_labels_func, contrastive_loss=None):
        """Compute total loss (flow + optional physics + contrastive). Returns (total_loss, loss_dict)."""
        loss_dict = {}
        base_flow_loss = self.base_criterion(predicted_flow, flow_batch)
        loss_dict['flow_loss'] = base_flow_loss.item()
        if True:
            batch_size = predicted_flow.shape[0]
            pred_fvc_list = []
            true_fvc_list = []
            pred_fev1_list = []
            true_fev1_list = []
            
            pred_flow_np = predicted_flow.detach().cpu().numpy()
            time_np = time_batch.cpu().numpy()
            
            for i in range(batch_size):
                pred_fvc, pred_fev1, pred_pef = calculate_fvc_fev1(pred_flow_np[i], time_np[i], method="integration")
                pred_fvc_list.append(pred_fvc)
                pred_fev1_list.append(pred_fev1)
                true_labels = get_true_labels_func(filenames[i])
                true_fvc = true_labels['fvc']
                true_fev1 = true_labels['fev1']
                true_fvc_list.append(true_fvc if true_fvc is not None else 0.0)
                true_fev1_list.append(true_fev1 if true_fev1 is not None else 0.0)
            pred_fvc_tensor = torch.tensor(pred_fvc_list, dtype=torch.float32, device=predicted_flow.device)
            true_fvc_tensor = torch.tensor(true_fvc_list, dtype=torch.float32, device=predicted_flow.device)
            pred_fev1_tensor = torch.tensor(pred_fev1_list, dtype=torch.float32, device=predicted_flow.device)
            true_fev1_tensor = torch.tensor(true_fev1_list, dtype=torch.float32, device=predicted_flow.device)
            time_batch_for_physics = time_batch
            if time_batch_for_physics.device != predicted_flow.device:
                time_batch_for_physics = time_batch_for_physics.to(predicted_flow.device)
            physics_loss, physics_loss_dict = self.physics_loss(
                    predicted_flow, flow_batch, 
                    pred_fvc_tensor, true_fvc_tensor,
                    pred_fev1_tensor, true_fev1_tensor,
                    time_batch_for_physics
                )
            loss_dict.update(physics_loss_dict)
            total_loss = physics_loss
        if contrastive_loss is not None:
            loss_dict['contrastive_loss'] = contrastive_loss.item()
            total_loss = total_loss + self.contrastive_weight * contrastive_loss
        else:
            loss_dict['contrastive_loss'] = 0.0
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def _calculate_ratio_loss(self, predicted_flow, time_batch, filenames, get_true_labels_func):
        """Compute FEV1/FVC ratio loss (legacy)."""
        batch_size = predicted_flow.shape[0]
        pred_ratios = []
        true_ratios = []
        pred_flow_np = predicted_flow.detach().cpu().numpy()
        time_np = time_batch.cpu().numpy()
        
        for i in range(batch_size):
            pred_fvc, pred_fev1, pred_pef = calculate_fvc_fev1(
                pred_flow_np[i], 
                time_np[i], 
                method="integration"
            )
            
            true_labels = get_true_labels_func(filenames[i])
            fvc_true = true_labels['fvc']
            fev1_true = true_labels['fev1']
            if fvc_true is not None and fev1_true is not None and fvc_true > 0 and pred_fvc > 0:
                pred_ratio = pred_fev1 / pred_fvc
                true_ratio = fev1_true / fvc_true
                
                pred_ratios.append(pred_ratio)
                true_ratios.append(true_ratio)
        if len(pred_ratios) > 0:
            pred_ratios_tensor = torch.tensor(pred_ratios, dtype=torch.float32, device=predicted_flow.device)
            true_ratios_tensor = torch.tensor(true_ratios, dtype=torch.float32, device=predicted_flow.device)
            ratio_loss = nn.L1Loss()(pred_ratios_tensor, true_ratios_tensor)
        else:
            ratio_loss = torch.tensor(0.0, device=predicted_flow.device)
        
        return ratio_loss
    
    def get_loss_info(self):
        """Return loss info."""
        return {
            'use_physics_loss': False,
            'use_fev1_weighted_loss': False,
            'loss_functions_enabled': ['PhysicsInformedLossV2', 'contrastive']
        }

