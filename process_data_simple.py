# -*- coding: utf-8 -*-
"""
Flow + mel preprocessing: interpolate flow to 0-3s uniform grid (60 points), (0,0)-(3,0) endpoints;
options: linear, pchip, akima, cubic_spline, cubic_spline_clamped, bspline. Mel frames aligned to labels.
"""
import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import json
from config import config as Config

try:
    from scipy.interpolate import (
        PchipInterpolator,
        Akima1DInterpolator,
        CubicSpline,
        interp1d,
        make_interp_spline
    )
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not installed; only linear interpolation available. pip install scipy for others.")


def interpolate_flow_data(time_data, flow_data, target_time, method='linear'):
    """Interpolate flow to target_time. method: linear, pchip, akima, cubic_spline, cubic_spline_clamped, bspline."""
    if len(time_data) < 2:
        return np.zeros_like(target_time)
    unique_indices = np.unique(time_data, return_index=True)[1]
    time_data = time_data[unique_indices]
    flow_data = flow_data[unique_indices]
    
    if len(time_data) < 2:
        return np.zeros_like(target_time)
    if method == 'linear':
        processed_flow = np.interp(target_time, time_data, flow_data)
    elif method == 'pchip':
        if not SCIPY_AVAILABLE:
            print("Warning: scipy not installed, using linear")
            processed_flow = np.interp(target_time, time_data, flow_data)
        else:
            if len(time_data) >= 3:
                pchip = PchipInterpolator(time_data, flow_data)
                processed_flow = pchip(target_time)
            else:
                processed_flow = np.interp(target_time, time_data, flow_data)
    elif method == 'akima':
        if not SCIPY_AVAILABLE:
            print("Warning: scipy not installed, using linear")
            processed_flow = np.interp(target_time, time_data, flow_data)
        else:
            if len(time_data) >= 3:
                akima = Akima1DInterpolator(time_data, flow_data)
                processed_flow = akima(target_time)
            else:
                processed_flow = np.interp(target_time, time_data, flow_data)
    
    elif method == 'cubic_spline':
        if not SCIPY_AVAILABLE:
            print("Warning: scipy not installed, using linear")
            processed_flow = np.interp(target_time, time_data, flow_data)
        else:
            if len(time_data) >= 4:
                cs = CubicSpline(time_data, flow_data, bc_type='natural')
                processed_flow = cs(target_time)
            else:
                processed_flow = np.interp(target_time, time_data, flow_data)
    elif method == 'cubic_spline_clamped':
        if not SCIPY_AVAILABLE:
            print("Warning: scipy not installed, using linear")
            processed_flow = np.interp(target_time, time_data, flow_data)
        else:
            if len(time_data) >= 4:
                cs = CubicSpline(time_data, flow_data, bc_type=((1, 0.0), (1, 0.0)))
                processed_flow = cs(target_time)
            else:
                processed_flow = np.interp(target_time, time_data, flow_data)
    
    elif method == 'bspline':
        if not SCIPY_AVAILABLE:
            print("Warning: scipy not installed, using linear")
            processed_flow = np.interp(target_time, time_data, flow_data)
        else:
            if len(time_data) >= 4:
                spl = make_interp_spline(time_data, flow_data, k=3)
                processed_flow = spl(target_time)
            else:
                processed_flow = np.interp(target_time, time_data, flow_data)
    
    else:
        print(f"Warning: unknown method '{method}', using linear")
        processed_flow = np.interp(target_time, time_data, flow_data)
    
    return processed_flow


def process_flow_data(flow_data, time_data, interpolation_method='linear', sequence_length=None):
    """Resample flow to 0-3s uniform grid with (0,0) and (3,0); optional interpolation method."""
    if sequence_length is None:
        sequence_length = Config.SEQUENCE_LENGTH
    
    target_time = np.linspace(0, 3.0, sequence_length)
    mask = (time_data >= 0.0) & (time_data <= 3.0)
    time_data = time_data[mask]
    flow_data = flow_data[mask]
    
    if len(flow_data) == 0:
        processed_flow = np.zeros(sequence_length)
        processed_time = target_time
        return processed_flow, processed_time
    
    if time_data[0] > 0:
        time_data = np.concatenate([[0.0], time_data])
        flow_data = np.concatenate([[0.0], flow_data])
    else:
        flow_data[0] = 0.0
    last_time = time_data[-1]
    last_flow = flow_data[-1]
    
    if last_time < 3.0:
        n_decay_points = max(5, int((3.0 - last_time) * 10))
        decay_time = np.linspace(last_time, 3.0, n_decay_points)[1:]
        decay_flow = np.linspace(last_flow, 0.0, n_decay_points)[1:]
        
        time_data = np.concatenate([time_data, decay_time])
        flow_data = np.concatenate([flow_data, decay_flow])
    
    sorted_indices = np.argsort(time_data)
    time_data = time_data[sorted_indices]
    flow_data = flow_data[sorted_indices]
    mask = time_data <= 3.0
    time_data = time_data[mask]
    flow_data = flow_data[mask]
    if time_data[-1] < 3.0:
        time_data = np.concatenate([time_data, [3.0]])
        flow_data = np.concatenate([flow_data, [0.0]])
    else:
        flow_data[-1] = 0.0
    processed_flow = interpolate_flow_data(
        time_data, 
        flow_data, 
        target_time, 
        method=interpolation_method
    )
    
    processed_flow[0] = 0.0
    processed_flow[-1] = 0.0
    
    processed_time = target_time
    
    return processed_flow, processed_time


def generate_mel_spectrogram(audio, target_frames=None):
    """Compute mel spectrogram (dB), fix length to target_frames. Not normalized."""
    if target_frames is None:
        target_frames = Config.MEL_TIME_FRAMES
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        fmax=Config.SAMPLE_RATE // 2
    )
    
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec = librosa.util.fix_length(mel_spec, size=target_frames, axis=1)
    return mel_spec


def process_dataset(input_wav_dir=None, input_csv_dir=None, output_dir=None, interpolation_method='linear', sequence_length=None):
    """Process all wav+csv: flow interpolation + mel; output_dir default data_aug_{method}."""
    if output_dir is None:
        output_dir = f"data_aug_{interpolation_method}"
    
    print("Processing dataset...")
    print("=" * 70)
    mel_dir = os.path.join(output_dir, "mel")
    csv_dir = os.path.join(output_dir, "csv")
    
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    
    print(f"Output: {output_dir}, mel: {mel_dir}, csv: {csv_dir}")
    wav_dir = input_wav_dir if input_wav_dir is not None else Config.WAV_DIR
    csv_original_dir = input_csv_dir if input_csv_dir is not None else Config.CSV_DIR
    
    if sequence_length is None:
        seq_len = Config.SEQUENCE_LENGTH
    else:
        seq_len = sequence_length
    print(f"WAV: {wav_dir}, CSV: {csv_original_dir}, method: {interpolation_method}, length: {seq_len}")
    wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith('.wav')])
    print(f"Found {len(wav_files)} wav files")
    print("=" * 70)
    stats = {
        'total_files': len(wav_files),
        'processed_files': 0,
        'failed_files': 0
    }
    
    for wav_file in tqdm(wav_files, desc="Processing"):
        try:
            wav_path = os.path.join(wav_dir, wav_file)
            base_name = wav_file.replace('.wav', '')
            csv_file = f"{base_name}.csv"
            csv_path = os.path.join(csv_original_dir, csv_file)
            if not os.path.exists(csv_path):
                subject_id = base_name.split('_')[0]
                csv_file_alt = f"flowdata-{subject_id}.csv"
                csv_path_alt = os.path.join(csv_original_dir, csv_file_alt)
                if os.path.exists(csv_path_alt):
                    csv_path = csv_path_alt
                    csv_file = csv_file_alt
                else:
                    print(f"Warning: no CSV for {csv_file} or {csv_file_alt}")
                    stats['failed_files'] += 1
                    continue
            audio, _ = librosa.load(wav_path, sr=Config.SAMPLE_RATE)
            flow_data = pd.read_csv(csv_path, header=None, names=['volume', 'flow', 'time'])
            flow_sequence = flow_data['flow'].values
            time_sequence = flow_data['time'].values
            
            if sequence_length is None:
                seq_len = Config.SEQUENCE_LENGTH
            else:
                seq_len = sequence_length
            processed_flow, processed_time = process_flow_data(
                flow_sequence, 
                time_sequence, 
                interpolation_method=interpolation_method,
                sequence_length=seq_len
            )
            
            mel_spec = generate_mel_spectrogram(audio, target_frames=seq_len)
            mel_path = os.path.join(mel_dir, f"{base_name}.npy")
            csv_output_path = os.path.join(csv_dir, f"{base_name}.csv")
            np.save(mel_path, mel_spec)
            output_df = pd.DataFrame({
                'time': processed_time,
                'flow': processed_flow
            })
            output_df.to_csv(csv_output_path, index=False, header=False)
            
            stats['processed_files'] += 1
            
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
            stats['failed_files'] += 1
    stats_path = os.path.join(output_dir, "processing_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 70)
    print("Done.")
    print(f"Total: {stats['total_files']}, Processed: {stats['processed_files']}, Failed: {stats['failed_files']}")
    print(f"Stats: {stats_path}, Data: {output_dir}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess flow + mel; optional interpolation method.')
    parser.add_argument('--wav_dir', type=str, default=None, help='Input WAV dir (default: Config.WAV_DIR)')
    parser.add_argument('--csv_dir', type=str, default=None, help='Input CSV dir (default: Config.CSV_DIR)')
    parser.add_argument('--method', type=str, default='linear',
                        choices=['linear', 'pchip', 'akima', 'cubic_spline', 'cubic_spline_clamped', 'bspline'],
                        help='Interpolation method')
    parser.add_argument('--output_dir', type=str, default=None, help='Output dir (default: data_aug_{method})')
    parser.add_argument('--sequence_length', type=int, default=None, help='Number of flow points (default: 60)')
    
    args = parser.parse_args()
    
    stats = process_dataset(
        input_wav_dir=args.wav_dir,
        input_csv_dir=args.csv_dir,
        output_dir=args.output_dir,
        interpolation_method=args.method,
        sequence_length=args.sequence_length
    )
    
    actual_output_dir = args.output_dir if args.output_dir else f"data_aug_{args.method}"
    print("\n" + "=" * 70)
    print("Done.")
    print(f"Method: {args.method}, Output: {actual_output_dir}")
    print(f"Processed: {stats['processed_files']}, Failed: {stats['failed_files']}")
