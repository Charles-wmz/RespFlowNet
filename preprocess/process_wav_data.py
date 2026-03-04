#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unify audio to fixed duration (e.g. 3s) and sample rate; pad or truncate."""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm


def unify_audio_length(audio, sr, target_duration=3.0):
    """Pad (linear decay to 0) or truncate to target_duration seconds."""
    target_length = int(sr * target_duration)
    current_length = len(audio)
    if current_length == target_length:
        return audio
    if current_length < target_length:
        pad_length = target_length - current_length
        fade_out = np.linspace(audio[-1], 0, pad_length)
        return np.concatenate([audio, fade_out])
    return audio[:target_length]


def process_audio_file(input_path, output_path, target_duration=3.0, target_sr=48000):
    """Load, resample if needed, unify length, save. Returns True on success."""
    try:
        audio, sr = librosa.load(input_path, sr=None)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        unified_audio = unify_audio_length(audio, sr, target_duration)
        sf.write(output_path, unified_audio, sr)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def unify_dataset_audio_dimensions(input_dir, output_dir, target_duration=3.0, target_sr=48000):
    """Batch process: resample and unify length for all wav in input_dir -> output_dir."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    audio_files = list(input_path.glob("*.wav"))
    
    if len(audio_files) == 0:
        print(f"No wav files in {input_dir}")
        return
    print(f"Found {len(audio_files)} files. Target: {target_sr}Hz, {target_duration}s. Output: {output_dir}")
    print("=" * 70)
    success_count = 0
    padded_count = 0
    truncated_count = 0
    exact_count = 0
    
    for audio_file in tqdm(audio_files, desc="Processing"):
        input_file_path = audio_file
        output_file_path = output_path / audio_file.name
        
        try:
            audio_original, sr_original = librosa.load(input_file_path, sr=None)
            original_length = len(audio_original)
            target_length = int(target_sr * target_duration)
            
            if original_length < target_length:
                padded_count += 1
            elif original_length > target_length:
                truncated_count += 1
            else:
                exact_count += 1
            
            if process_audio_file(input_file_path, output_file_path, target_duration, target_sr):
                success_count += 1
                
        except Exception as e:
            print(f"\nError {audio_file.name}: {e}")
    print("\n" + "=" * 70)
    print("Done.")
    print(f"Total: {len(audio_files)}, Success: {success_count}, Failed: {len(audio_files) - success_count}")
    print(f"Padded: {padded_count}, Truncated: {truncated_count}, Unchanged: {exact_count}")
    print("=" * 70)


def verify_output(output_dir, target_duration=3.0, target_sr=48000, num_samples=5):
    """Spot-check output files for correct duration and sample rate."""
    output_path = Path(output_dir)
    audio_files = list(output_path.glob("*.wav"))
    
    if len(audio_files) == 0:
        print(f"No audio in {output_dir}")
        return
    print("\nVerifying output...")
    print("=" * 70)
    sample_files = np.random.choice(audio_files, min(num_samples, len(audio_files)), replace=False)
    
    target_length = int(target_sr * target_duration)
    all_correct = True
    
    for audio_file in sample_files:
        audio, sr = librosa.load(audio_file, sr=None)
        duration = len(audio) / sr
        
        is_correct = (sr == target_sr and len(audio) == target_length)
        status = "[OK]" if is_correct else "[ERROR]"
        
        print(f"{status} {audio_file.name}: {sr}Hz, {len(audio)} samples, {duration:.3f}s")
        
        if not is_correct:
            all_correct = False
    
    print("=" * 70)
    if all_correct:
        print("Verification passed.")
    else:
        print("Warning: some files do not match target spec.")
    print("=" * 70)


if __name__ == "__main__":
    INPUT_DIR = "../New_data/wav_0"
    OUTPUT_DIR = "../New_data/wav"
    TARGET_DURATION = 3.0
    TARGET_SR = 48000
    print("Audio preprocessing tool")
    print("=" * 70)
    print(f"Input: {INPUT_DIR}, Output: {OUTPUT_DIR}, {TARGET_SR}Hz, {TARGET_DURATION}s")
    print("=" * 70)
    unify_dataset_audio_dimensions(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        target_duration=TARGET_DURATION,
        target_sr=TARGET_SR
    )
    verify_output(
        output_dir=OUTPUT_DIR,
        target_duration=TARGET_DURATION,
        target_sr=TARGET_SR,
        num_samples=10
    )


