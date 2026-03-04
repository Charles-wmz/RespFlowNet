# Respiratory Flow Prediction with CNN-LSTM

This repository contains a CNN-LSTM pipeline for predicting respiratory flow curves from audio (mel spectrograms). The model uses a contrastive gender encoder, dynamic memory enhancement, and physics-informed loss (FVC/FEV1 integral consistency). It supports 5-fold cross-validation and derives FVC/FEV1/PEF from predicted flow via integration.

## Datasets

Due to data privacy and protection protocols, we cannot disclose all the data. We are providing 10 samples for reference only.

## Features

- **Full model**: Contrastive learning gender encoder + dynamic memory network + gender input
- **Loss**: Flow MAE + physics-informed (FVC/FEV1 integral + smoothness) + contrastive
- **5-fold cross-validation** with GroupKFold by subject
- **Metrics**: Flow MAE/RMSE/MAPE/R²; FVC, FEV1, PEF, FEV1/FVC MAE/RMSE/MAPE/ICC
- **Preprocessing**: F-V curve cleaning (CSV), audio normalization (WAV), flow+mel (configurable interpolation)

## Requirements

- Python 3.8+
- PyTorch, numpy, pandas, scipy, librosa, scikit-learn, tqdm, pyyaml
- Optional: tensorboard, matplotlib

## Data Layout

- **Labels**: CSV with columns `id`, `fvc`, `fev1`, `gender` (e.g. `Lung_fun_label.csv`). Subject ID is the first segment of `id` (e.g. `0001` from `0001_2310714_20250320`).
- **Training data**: Mel spectrograms in `./data_aug/mel` (`.npy`) and flow CSV in `./data_aug/csv` (columns: time, flow). Audio 0–3 s; flow resampled to 60 points, (0,0)–(3,0).
- **Paths**: Set in `config.yaml`.

## Preprocessing

1. **F-V CSV** (optional): `python preprocess/process_csv_data.py` — edit `INPUT_DIR`, `OUTPUT_DIR`, `FILE_PATTERN` in file.
2. **Audio** (optional): `python preprocess/process_wav_data.py` — adjust input/output and target duration/SR in `__main__`.
3. **Flow + Mel** (for training):  
   `python process_data_simple.py --wav_dir <wav_dir> --csv_dir <csv_dir> --method linear [--output_dir <dir>]`  
   Use the same output dir as in `config.yaml` (e.g. `data_aug`).

## Training

```bash
python run_cross_modular.py --experiment run_name
python run_cross_modular.py --config training.epochs=100 preprocessing.batch_size=16
```

Results under `Config.OUTPUT_DIR` (e.g. `output/run_name/`): per-fold models, logs, validation CSVs, `cross_validation_results/cross_validation_results.json` and `.txt`.

## Config

- `config.yaml`: data paths, audio/preprocessing/model/training parameters.
- Override at runtime: `--config key=value` (e.g. `training.epochs=50`).

## Project Structure

- `run_cross_modular.py`: 5-fold CV entry; builds full model and trainer
- `config.py`: Load YAML + CLI overrides
- `model_modular.py`: Full CNN-LSTM (contrastive gender encoder + dynamic memory); `create_model()`
- `trainer_modular.py`: Loss = flow + physics v2 + contrastive
- `cross_validation_trainer.py`: Base 5-fold trainer (metrics, checkpointing)
- `cross_validation_data_loader.py`: GroupKFold data loader with gender
- `metrics.py`: FVC/FEV1/PEF from flow, ICC, flow/FVC/FEV1 metrics
- `modules/`: Contrastive gender encoder, physics loss v2, dynamic memory network
- `preprocess/`: F-V CSV cleaning, audio normalization
- `process_data_simple.py`: Flow interpolation + mel for training data

## License

Use and modify as needed. If you use this code in research, please cite the associated paper or project.

