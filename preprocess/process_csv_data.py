"""
F-V curve data preprocessing: Excel to CSV, cleaning (completeness, negatives, sort, dedup, monotonicity), time column via trapezoidal rule. Output: Volume, Flow, Time.
Configure INPUT_DIR, OUTPUT_DIR, FILE_PATTERN at top. Run: python process_csv_data.py
"""

import os
import csv
import glob
import shutil
import pandas as pd
from pathlib import Path
import logging

INPUT_DIR = '../New_data/csv_0'
OUTPUT_DIR = '../New_data/csv'
FILE_PATTERN = 'flowdata-*.csv'
VERBOSE = True
SHOW_NO_ISSUE_FILES = False
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_excel_files():
    """Convert xlsx to CSV (skip first 2 rows)."""
    input_dir = Path(INPUT_DIR)
    temp_output_dir = Path(OUTPUT_DIR)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return False
    temp_output_dir.mkdir(parents=True, exist_ok=True)
    xlsx_files = list(input_dir.glob('*.xlsx'))
    if not xlsx_files:
        logger.info(f"No xlsx in {input_dir}, skipping Excel step")
        return True
    logger.info(f"Found {len(xlsx_files)} xlsx files")
    
    processed_count = 0
    error_count = 0
    
    for xlsx_file in xlsx_files:
        try:
            logger.info(f"Processing: {xlsx_file.name}")
            df = pd.read_excel(xlsx_file, skiprows=2)
            original_rows = len(df)
            logger.info(f"Rows: {original_rows}")
            if original_rows == 0:
                logger.warning(f"{xlsx_file.name} has no data after skip, skipping")
                continue
            df_processed = df.reset_index(drop=True)
            processed_rows = len(df_processed)
            output_filename = xlsx_file.stem + '.csv'
            output_path = temp_output_dir / output_filename
            df_processed.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"Saved: {output_path}")
            
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {xlsx_file.name}: {e}")
            error_count += 1
            continue
    logger.info("=" * 50)
    logger.info("Excel processing done.")
    logger.info(f"Success: {processed_count}, Failed: {error_count}, Total: {len(xlsx_files)}")
    logger.info("=" * 50)
    
    return error_count == 0

def calculate_time_column(volumes, flows):
    """Compute time column via trapezoidal rule. t0 = V0/F0; t_n = t_{n-1} + dV / avg_flow."""
    if len(volumes) == 0 or len(flows) == 0:
        return []
    
    times = []
    
    if flows[0] != 0:
        t0 = volumes[0] / flows[0]
    else:
        t0 = 0.0
    times.append(t0)
    for i in range(1, len(volumes)):
        delta_v = volumes[i] - volumes[i-1]
        avg_flow = (flows[i] + flows[i-1]) / 2.0
        delta_t = delta_v / avg_flow if avg_flow != 0 else 0.0
        
        t_n = times[i-1] + delta_t
        times.append(t_n)
    
    return times

def process_csv_file_cleaning(input_file, output_file, verbose=True):
    """Clean CSV: drop incomplete/negative, sort by volume, dedup, enforce monotonicity. Returns stats dict."""
    raw_volumes = []
    raw_flows = []
    original_line_numbers = []
    
    line_num = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_num += 1
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) >= 1:
                    try:
                        volume = float(parts[0]) if parts[0] else None
                        flow = float(parts[1]) if len(parts) > 1 and parts[1].strip() else None
                        raw_volumes.append(volume)
                        raw_flows.append(flow)
                        original_line_numbers.append(line_num)
                    except ValueError:
                        continue
    
    total_lines = len(raw_volumes)
    
    volumes = []
    flows = []
    valid_line_numbers = []
    incomplete_lines = []
    negative_lines = []
    
    for i in range(len(raw_volumes)):
        volume = raw_volumes[i]
        flow = raw_flows[i]
        
        if volume is None or flow is None:
            incomplete_lines.append(original_line_numbers[i])
            continue
        
        if volume < 0 or flow < 0:
            negative_lines.append(original_line_numbers[i])
            continue
        volumes.append(volume)
        flows.append(flow)
        valid_line_numbers.append(original_line_numbers[i])
    
    lines_removed_incomplete = len(incomplete_lines)
    lines_removed_negative = len(negative_lines)
    
    sorted_indices = sorted(range(len(volumes)), key=lambda i: volumes[i])
    sorted_volumes = [volumes[i] for i in sorted_indices]
    sorted_flows = [flows[i] for i in sorted_indices]
    sorted_line_numbers = [valid_line_numbers[i] for i in sorted_indices]
    
    unique_volumes = []
    unique_flows = []
    unique_line_numbers = []
    duplicate_lines = []
    
    seen_volumes = set()
    for i in range(len(sorted_volumes)):
        volume = sorted_volumes[i]
        if volume not in seen_volumes:
            seen_volumes.add(volume)
            unique_volumes.append(volume)
            unique_flows.append(sorted_flows[i])
            unique_line_numbers.append(sorted_line_numbers[i])
        else:
            duplicate_lines.append(sorted_line_numbers[i])
    
    lines_removed_duplicate = len(duplicate_lines)
    cutoff_index = len(unique_volumes)
    monotonicity_issue_line = 0
    
    for i in range(1, len(unique_volumes)):
        if unique_volumes[i] < unique_volumes[i-1]:
            cutoff_index = i
            monotonicity_issue_line = unique_line_numbers[i]
            break
    
    lines_removed_monotonicity = len(unique_volumes) - cutoff_index
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        for i in range(cutoff_index):
            f.write(f"{unique_volumes[i]},{unique_flows[i]}\n")
    return {
        'total_lines': total_lines,
        'incomplete_lines': incomplete_lines,
        'negative_lines': negative_lines,
        'duplicate_lines': duplicate_lines,
        'monotonicity_issue_line': monotonicity_issue_line,
        'lines_removed_incomplete': lines_removed_incomplete,
        'lines_removed_negative': lines_removed_negative,
        'lines_removed_duplicate': lines_removed_duplicate,
        'lines_removed_monotonicity': lines_removed_monotonicity,
        'final_lines': cutoff_index
    }


def process_csv_file_time_column(input_file, output_file, verbose=True):
    """Add time column to cleaned CSV. Returns stats dict."""
    volumes = []
    flows = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        volume = float(parts[0])
                        flow = float(parts[1])
                        volumes.append(volume)
                        flows.append(flow)
                    except ValueError:
                        continue
    
    times = calculate_time_column(volumes, flows)
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        for i in range(len(volumes)):
            f.write(f"{volumes[i]},{flows[i]},{times[i]}\n")
    
    return {
        'data_points': len(volumes),
        't0': times[0] if len(times) > 0 else 0,
        't_final': times[-1] if len(times) > 0 else 0,
        'duration': times[-1] - times[0] if len(times) > 0 else 0
    }


def main():
    print("=" * 80)
    print("               F-V Curve Data Preprocessing")
    print("=" * 80)
    print()
    logger.info("Processing Excel (skip 2 rows, convert to CSV)...")
    
    try:
        excel_success = process_excel_files()
        if not excel_success:
            logger.error("Excel processing failed, aborting")
            return
    except Exception as e:
        logger.error(f"Excel error: {e}")
        return
    logger.info("Excel done. Starting CSV cleaning...")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[+] Created: {OUTPUT_DIR}")
    file_pattern = os.path.join(OUTPUT_DIR, FILE_PATTERN)
    csv_files = glob.glob(file_pattern)
    if not csv_files:
        print(f"[!] No files matching '{FILE_PATTERN}' in '{OUTPUT_DIR}'")
        return
    total_files = len(csv_files)
    print(f"[+] Found {total_files} files to process")
    print(f"[+] Directory: {OUTPUT_DIR}")
    print()
    print("-" * 80)
    print()
    
    print("Step 1: Data cleaning...")
    files_with_incomplete = 0
    files_with_negative = 0
    files_with_duplicate = 0
    files_with_monotonicity_issue = 0
    total_lines_removed_incomplete = 0
    total_lines_removed_negative = 0
    total_lines_removed_duplicate = 0
    total_lines_removed_monotonicity = 0
    
    temp_dir = os.path.join(OUTPUT_DIR, 'temp_cleaned')
    os.makedirs(temp_dir, exist_ok=True)
    for csv_file in sorted(csv_files):
        filename = os.path.basename(csv_file)
        temp_output_file = os.path.join(temp_dir, filename)
        
        stats = process_csv_file_cleaning(csv_file, temp_output_file, verbose=VERBOSE)
        has_issues = False
        if stats['lines_removed_incomplete'] > 0:
            files_with_incomplete += 1
            total_lines_removed_incomplete += stats['lines_removed_incomplete']
            has_issues = True
        
        if stats['lines_removed_negative'] > 0:
            files_with_negative += 1
            total_lines_removed_negative += stats['lines_removed_negative']
            has_issues = True
        
        if stats['lines_removed_duplicate'] > 0:
            files_with_duplicate += 1
            total_lines_removed_duplicate += stats['lines_removed_duplicate']
            has_issues = True
        
        if stats['lines_removed_monotonicity'] > 0:
            files_with_monotonicity_issue += 1
            total_lines_removed_monotonicity += stats['lines_removed_monotonicity']
            has_issues = True
        
        if VERBOSE and has_issues:
            print(f"[!] {filename}")
            print(f"    total_lines: {stats['total_lines']}")
            if stats['lines_removed_incomplete'] > 0:
                print(f"    incomplete: removed {stats['lines_removed_incomplete']} rows")
            if stats['lines_removed_negative'] > 0:
                print(f"    negative: removed {stats['lines_removed_negative']} rows")
            if stats['lines_removed_duplicate'] > 0:
                print(f"    duplicate: removed {stats['lines_removed_duplicate']} rows")
            if stats['lines_removed_monotonicity'] > 0:
                print(f"    monotonicity: from line {stats['monotonicity_issue_line']} removed {stats['lines_removed_monotonicity']} rows")
            print(f"    final: {stats['final_lines']} rows")
            print()
        elif VERBOSE and SHOW_NO_ISSUE_FILES:
            print(f"[OK] {filename} - no issues ({stats['final_lines']} rows)")
    print("\nStep 2: Adding time column...")
    temp_csv_files = glob.glob(os.path.join(temp_dir, FILE_PATTERN))
    
    for temp_csv_file in sorted(temp_csv_files):
        filename = os.path.basename(temp_csv_file)
        final_output_file = os.path.join(OUTPUT_DIR, filename)
        time_stats = process_csv_file_time_column(temp_csv_file, final_output_file, verbose=VERBOSE)
        if VERBOSE:
            print(f"[OK] {filename} points={time_stats['data_points']} t0={time_stats['t0']:.6f}s duration={time_stats['duration']:.6f}s")
            print()
    import shutil
    shutil.rmtree(temp_dir)
    print("-" * 80)
    print("Done.")
    print(f"Summary: files={total_files}, incomplete={files_with_incomplete}, negative={files_with_negative}, duplicate={files_with_duplicate}, monotonicity={files_with_monotonicity_issue}")
    print(f"Lines removed: incomplete={total_lines_removed_incomplete}, negative={total_lines_removed_negative}, duplicate={total_lines_removed_duplicate}, monotonicity={total_lines_removed_monotonicity}")
    print(f"[+] Output: {OUTPUT_DIR} (Volume,Flow,Time)")
    print("=" * 80)


if __name__ == "__main__":
    main()
