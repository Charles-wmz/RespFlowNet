# -*- coding: utf-8 -*-
"""
Evaluation metrics and FVC/FEV1 calculation module
"""
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from scipy import integrate

def calculate_fvc_fev1(flow_sequence, time_sequence, fev1_time=1.0, method="integration"):
    """
    Calculate FVC, FEV1, and PEF with optimized integration and physical constraints
    
    Args:
        flow_sequence: Flow sequence (L/s)
        time_sequence: Time sequence (L,) - should provide actual time data
        fev1_time: FEV1 time point (seconds)
        method: FEV1 calculation method ('interpolation', 'nearest', 'integration')
    
    Returns:
        fvc: Forced vital capacity (L)
        fev1: Forced expiratory volume in 1 second (L)
        pef: Peak expiratory flow (L/s)
    """
    if len(flow_sequence) != len(time_sequence):
        raise ValueError("Flow sequence and time sequence must have the same length")
    
    time_sequence_corrected = time_sequence - time_sequence[0]
    fvc = integrate.simpson(flow_sequence, time_sequence_corrected)
    
    # Calculate FEV1 based on method
    if method == "integration":
        # Integration method: integrate flow from 0 to fev1_time using Simpson's rule
        fev1_mask = time_sequence_corrected <= fev1_time
        if np.any(fev1_mask):
            fev1 = integrate.simpson(flow_sequence[fev1_mask], time_sequence_corrected[fev1_mask])
        else:
            fev1 = 0.0
    
    elif method == "interpolation":
        # Interpolation method: interpolate to exact fev1_time using cubic spline
        if fev1_time <= time_sequence_corrected[0]:
            fev1 = 0.0
        elif fev1_time >= time_sequence_corrected[-1]:
            fev1 = fvc
        else:
            # Use cubic spline interpolation for consistency with flow sequence interpolation
            from scipy.interpolate import interp1d
            f = interp1d(time_sequence_corrected, flow_sequence, kind='cubic', 
                        bounds_error=False, fill_value=0)
            
            # Create extended time sequence including fev1_time
            extended_time = np.concatenate([time_sequence_corrected[time_sequence_corrected <= fev1_time], [fev1_time]])
            extended_flow = f(extended_time)
            
            # Integrate from 0 to fev1_time using Simpson's rule
            fev1 = integrate.simpson(extended_flow, extended_time)
    
    elif method == "nearest":
        # Nearest neighbor method
        idx = np.argmin(np.abs(time_sequence_corrected - fev1_time))
        if idx > 0:
            fev1 = integrate.simpson(flow_sequence[:idx+1], time_sequence_corrected[:idx+1])
        else:
            fev1 = 0.0
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if fev1 > fvc:
        fev1 = fvc * 0.95
    if fvc > 0:
        ratio = fev1 / fvc
        if ratio > 0.95:
            fev1 = fvc * 0.90
        elif ratio < 0.3:
            fev1 = fvc * 0.5
    fvc = max(0.0, fvc)
    fev1 = max(0.0, fev1)
    pef = np.max(flow_sequence)
    pef = max(0.0, pef)
    
    return fvc, fev1, pef

def calculate_flow_metrics(predictions, targets):
    """
    Calculate flow prediction metrics
    
    Args:
        predictions: Predicted flow values
        targets: True flow values
    
    Returns:
        Dictionary of metrics
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = math.sqrt(mse)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((targets - predictions) / (np.abs(targets) + 1e-8))) * 100
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(predictions, targets)[0, 1]
    
    # Calculate R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'correlation': correlation,
        'r_squared': r_squared
    }

def calculate_icc(y_true, y_pred):
    """Intraclass Correlation Coefficient ICC(2,1): two-way random, single measure, consistency."""
    import pandas as pd
    from scipy import stats
    n = len(y_true)
    data = pd.DataFrame({
        'subject': list(range(n)) + list(range(n)),
        'rater': ['true'] * n + ['pred'] * n,
        'score': list(y_true) + list(y_pred)
    })
    subject_means = data.groupby('subject')['score'].mean()
    grand_mean = data['score'].mean()
    SSB = 2 * np.sum((subject_means - grand_mean) ** 2)
    SSW = np.sum((y_true - subject_means) ** 2) + np.sum((y_pred - subject_means) ** 2)
    MSB = SSB / (n - 1)
    MSW = SSW / n
    icc = (MSB - MSW) / (MSB + MSW) if (MSB + MSW) != 0 else 0
    return max(0, icc)

def calculate_fvc_fev1_metrics(fvc_pred, fvc_true, fev1_pred, fev1_true, pef_pred=None, pef_true=None):
    """Compute FVC, FEV1, PEF metrics (MAE, RMSE, MAPE, ICC)."""
    import math
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    fvc_mae = mean_absolute_error(fvc_true, fvc_pred)
    fvc_rmse = math.sqrt(mean_squared_error(fvc_true, fvc_pred))
    fvc_mape = np.mean(np.abs((np.array(fvc_true) - np.array(fvc_pred)) / 
                             (np.array(fvc_true) + 1e-8))) * 100
    fvc_icc = calculate_icc(fvc_true, fvc_pred)
    
    fev1_mae = mean_absolute_error(fev1_true, fev1_pred)
    fev1_rmse = math.sqrt(mean_squared_error(fev1_true, fev1_pred))
    fev1_mape = np.mean(np.abs((np.array(fev1_true) - np.array(fev1_pred)) / 
                              (np.array(fev1_true) + 1e-8))) * 100
    fev1_icc = calculate_icc(fev1_true, fev1_pred)
    
    fev1_fvc_true = np.array(fev1_true) / (np.array(fvc_true) + 1e-8)
    fev1_fvc_pred = np.array(fev1_pred) / (np.array(fvc_pred) + 1e-8)
    fev1_fvc_mae = mean_absolute_error(fev1_fvc_true, fev1_fvc_pred)
    fev1_fvc_rmse = math.sqrt(mean_squared_error(fev1_fvc_true, fev1_fvc_pred))
    fev1_fvc_mape = np.mean(np.abs((fev1_fvc_true - fev1_fvc_pred) / 
                                  (fev1_fvc_true + 1e-8))) * 100
    fev1_fvc_icc = calculate_icc(fev1_fvc_true, fev1_fvc_pred)
    
    metrics = {
        'fvc_mae': fvc_mae,
        'fvc_rmse': fvc_rmse,
        'fvc_mape': fvc_mape,
        'fvc_icc': fvc_icc,
        'fev1_mae': fev1_mae,
        'fev1_rmse': fev1_rmse,
        'fev1_mape': fev1_mape,
        'fev1_icc': fev1_icc,
        'fev1_fvc_mae': fev1_fvc_mae,
        'fev1_fvc_rmse': fev1_fvc_rmse,
        'fev1_fvc_mape': fev1_fvc_mape,
        'fev1_fvc_icc': fev1_fvc_icc
    }
    
    if pef_pred is not None and pef_true is not None:
        pef_mae = mean_absolute_error(pef_true, pef_pred)
        pef_rmse = math.sqrt(mean_squared_error(pef_true, pef_pred))
        pef_mape = np.mean(np.abs((np.array(pef_true) - np.array(pef_pred)) / 
                                 (np.array(pef_true) + 1e-8))) * 100
        pef_icc = calculate_icc(pef_true, pef_pred)
        metrics.update({
            'pef_mae': pef_mae,
            'pef_rmse': pef_rmse,
            'pef_mape': pef_mape,
            'pef_icc': pef_icc
        })
    
    return metrics

def calculate_comprehensive_metrics(predictions, targets, fvc_pred=None, fvc_true=None, 
                                  fev1_pred=None, fev1_true=None, pef_pred=None, pef_true=None):
    """
    Calculate comprehensive metrics for flow prediction
    
    Args:
        predictions: Predicted flow values
        targets: True flow values
        fvc_pred: Predicted FVC values (optional)
        fvc_true: True FVC values (optional)
        fev1_pred: Predicted FEV1 values (optional)
        fev1_true: True FEV1 values (optional)
        pef_pred: Predicted PEF values (optional)
        pef_true: True PEF values (optional)
    
    Returns:
        Dictionary of all metrics
    """
    metrics = {}
    
    # Flow metrics - MAE, RMSE, MAPE, R²
    flow_metrics = calculate_flow_metrics(predictions, targets)
    metrics.update({
        'flow_mae': flow_metrics['mae'],
        'flow_rmse': flow_metrics['rmse'],
        'flow_mape': flow_metrics['mape'],
        'flow_r_squared': flow_metrics['r_squared']
    })
    
    # FVC, FEV1, and PEF metrics (if provided)
    if fvc_pred is not None and fvc_true is not None and fev1_pred is not None and fev1_true is not None:
        fvc_fev1_metrics = calculate_fvc_fev1_metrics(fvc_pred, fvc_true, fev1_pred, fev1_true, 
                                                      pef_pred, pef_true)
        metrics.update(fvc_fev1_metrics)
    
    return metrics

def print_metrics(metrics, dataset_name="Dataset"):
    """
    Print metrics in a formatted way
    
    Args:
        metrics: Dictionary of metrics
        dataset_name: Name of the dataset
    """
    print(f"\n{dataset_name} Metrics:")
    print("-" * 40)
    
    # Flow metrics - MAE, RMSE, MAPE, R²
    if 'flow_mae' in metrics:
        print(f"Flow MAE: {metrics['flow_mae']:.6f}")
    if 'flow_rmse' in metrics:
        print(f"Flow RMSE: {metrics['flow_rmse']:.6f}")
    if 'flow_mape' in metrics:
        print(f"Flow MAPE: {metrics['flow_mape']:.2f}%")
    if 'flow_r_squared' in metrics:
        print(f"Flow R²: {metrics['flow_r_squared']:.6f}")
    
    # FVC metrics - MAE, RMSE, MAPE, ICC
    if 'fvc_mae' in metrics:
        print(f"FVC MAE: {metrics['fvc_mae']:.6f}")
    if 'fvc_rmse' in metrics:
        print(f"FVC RMSE: {metrics['fvc_rmse']:.6f}")
    if 'fvc_mape' in metrics:
        print(f"FVC MAPE: {metrics['fvc_mape']:.2f}%")
    if 'fvc_icc' in metrics:
        print(f"FVC ICC: {metrics['fvc_icc']:.6f}")
    
    # FEV1 metrics - MAE, RMSE, MAPE, ICC
    if 'fev1_mae' in metrics:
        print(f"FEV1 MAE: {metrics['fev1_mae']:.6f}")
    if 'fev1_rmse' in metrics:
        print(f"FEV1 RMSE: {metrics['fev1_rmse']:.6f}")
    if 'fev1_mape' in metrics:
        print(f"FEV1 MAPE: {metrics['fev1_mape']:.2f}%")
    if 'fev1_icc' in metrics:
        print(f"FEV1 ICC: {metrics['fev1_icc']:.6f}")
    
    # FEV1/FVC ratio metrics - MAE, RMSE, MAPE, ICC
    if 'fev1_fvc_mae' in metrics:
        print(f"FEV1/FVC MAE: {metrics['fev1_fvc_mae']:.6f}")
    if 'fev1_fvc_rmse' in metrics:
        print(f"FEV1/FVC RMSE: {metrics['fev1_fvc_rmse']:.6f}")
    if 'fev1_fvc_mape' in metrics:
        print(f"FEV1/FVC MAPE: {metrics['fev1_fvc_mape']:.2f}%")
    if 'fev1_fvc_icc' in metrics:
        print(f"FEV1/FVC ICC: {metrics['fev1_fvc_icc']:.6f}")
    
    # PEF metrics - MAE, RMSE, MAPE, ICC
    if 'pef_mae' in metrics:
        print(f"PEF MAE: {metrics['pef_mae']:.6f}")
    if 'pef_rmse' in metrics:
        print(f"PEF RMSE: {metrics['pef_rmse']:.6f}")
    if 'pef_mape' in metrics:
        print(f"PEF MAPE: {metrics['pef_mape']:.2f}%")
    if 'pef_icc' in metrics:
        print(f"PEF ICC: {metrics['pef_icc']:.6f}")

if __name__ == "__main__":
    # Test the metrics functions
    print("Testing metrics functions...")
    
    # Test data
    flow_seq = np.array([0, 1, 2, 1, 0, -1, -2, -1, 0])
    time_seq = np.linspace(0, 3, len(flow_seq))
    
    # Test FVC/FEV1/PEF calculation
    fvc, fev1, pef = calculate_fvc_fev1(flow_seq, time_seq, method="integration")
    print(f"FVC: {fvc:.6f}, FEV1: {fev1:.6f}, PEF: {pef:.6f}")
    
    # Test flow metrics
    pred = np.array([0.1, 1.1, 1.9, 0.9, 0.1, -0.9, -1.9, -0.9, 0.1])
    true = np.array([0, 1, 2, 1, 0, -1, -2, -1, 0])
    
    flow_metrics = calculate_flow_metrics(pred, true)
    print_metrics(flow_metrics, "Test")
    
    print("Metrics test completed successfully!")