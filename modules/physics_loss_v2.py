# -*- coding: utf-8 -*-
"""Physics-informed loss v2: flow + FVC/FEV1 integral consistency + smoothness; optional module."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsInformedLossV2(nn.Module):
    """Flow MAE + FVC integral (full) + FEV1 integral (0-1s) + smoothness (second-order diff)."""
    def __init__(self, 
                 flow_weight=2.0, 
                 fvc_integral_weight=0.8, 
                 fev1_integral_weight=1.0, 
                 smoothness_weight=0.3):
        super(PhysicsInformedLossV2, self).__init__()
        self.flow_weight = flow_weight
        self.fvc_integral_weight = fvc_integral_weight
        self.fev1_integral_weight = fev1_integral_weight
        self.smoothness_weight = smoothness_weight
        
    def forward(self, pred_flow, true_flow, pred_fvc, true_fvc, pred_fev1, true_fev1, time_seq):
        """Returns (total_loss, loss_dict)."""
        if time_seq.device != pred_flow.device:
            time_seq = time_seq.to(pred_flow.device)
        
        flow_loss = F.l1_loss(pred_flow, true_flow)
        pred_fvc_calc = []
        for i in range(pred_flow.size(0)):
            time_corrected = time_seq[i] - time_seq[i][0]
            fvc = torch.trapz(pred_flow[i], time_corrected)
            pred_fvc_calc.append(fvc)
        pred_fvc_calc = torch.stack(pred_fvc_calc)
        if true_fvc.device != pred_flow.device:
            true_fvc = true_fvc.to(pred_flow.device)
        if pred_fvc_calc.device != pred_flow.device:
            pred_fvc_calc = pred_fvc_calc.to(pred_flow.device)
        
        fvc_integral_loss = F.l1_loss(pred_fvc_calc, true_fvc)
        pred_fev1_calc = []
        for i in range(pred_flow.size(0)):
            time_corrected = time_seq[i] - time_seq[i][0]
            fev1_mask = time_corrected <= 1.0
            if fev1_mask.sum() > 0:
                fev1 = torch.trapz(pred_flow[i][fev1_mask], time_corrected[fev1_mask])
            else:
                fev1 = torch.tensor(0.0, device=pred_flow.device)
            pred_fev1_calc.append(fev1)
        pred_fev1_calc = torch.stack(pred_fev1_calc)
        if true_fev1.device != pred_flow.device:
            true_fev1 = true_fev1.to(pred_flow.device)
        if pred_fev1_calc.device != pred_flow.device:
            pred_fev1_calc = pred_fev1_calc.to(pred_flow.device)
        
        fev1_integral_loss = F.l1_loss(pred_fev1_calc, true_fev1)
        flow_diff1 = pred_flow[:, 1:] - pred_flow[:, :-1]
        flow_diff2 = flow_diff1[:, 1:] - flow_diff1[:, :-1]
        smoothness_loss = torch.mean(torch.abs(flow_diff2))
        total_loss = (self.flow_weight * flow_loss + 
                     self.fvc_integral_weight * fvc_integral_loss +
                     self.fev1_integral_weight * fev1_integral_loss +
                     self.smoothness_weight * smoothness_loss)
        loss_dict = {
            'flow_loss': flow_loss.item(),
            'fvc_integral_loss': fvc_integral_loss.item(),
            'fev1_integral_loss': fev1_integral_loss.item(),
            'smoothness_loss': smoothness_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict


class PhysicsInformedLossV2_Compat(nn.Module):
    """Backward-compatible wrapper: optional FEV1 args default to zeros."""
    def __init__(self, 
                 flow_weight=2.0, 
                 fvc_integral_weight=0.8,
                 fev1_integral_weight=1.0,
                 smoothness_weight=0.3):
        super(PhysicsInformedLossV2_Compat, self).__init__()
        self.v2_loss = PhysicsInformedLossV2(
            flow_weight=flow_weight,
            fvc_integral_weight=fvc_integral_weight,
            fev1_integral_weight=fev1_integral_weight,
            smoothness_weight=smoothness_weight
        )
    
    def forward(self, pred_flow, true_flow, pred_fvc, true_fvc, time_seq, 
                pred_fev1=None, true_fev1=None):
        if pred_fev1 is None or true_fev1 is None:
            batch_size = pred_flow.size(0)
            device = pred_flow.device
            pred_fev1 = torch.zeros(batch_size, device=device)
            true_fev1 = torch.zeros(batch_size, device=device)
            self.v2_loss.fev1_integral_weight = 0.0
        
        return self.v2_loss(pred_flow, true_flow, pred_fvc, true_fvc, 
                          pred_fev1, true_fev1, time_seq)

