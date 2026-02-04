# Neuro-AI (Brain → VLM)

Minimal pipelines to train linear ridge decoders mapping fMRI betas to CLIP image/text embeddings.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Place datasets under:  
_(requires installing some NSD files, will update that later)_

```bash
data/
```

(ignored by git)

Then run:

```bash
bash setup_data.sh
```

## Train Linear Decoders

### Run full pipeline

```bash
RUN_ID=exp_001 bash train_linear_decoders.sh
```

This performs:
- fMRI preprocessing
- CLIP embedding extraction
- dataset splits
- image + text ridge training
- multimodal prediction

All outputs are organized under a single run directory:

```
results/linear_decoders/{subject}/run_{RUN_ID}/
├── models/
│   ├── fmri_to_clip_image.pkl
│   ├── fmri_norm_stats_image.npz
│   ├── fmri_to_clip_text.pkl
│   └── fmri_norm_stats_text.npz
├── embeddings/
│   ├── image.npy
│   ├── text.npy
│   └── trial_indices.npy
└── results.json
```

Where:
- `models/` contains trained ridge decoders and normalization statistics
- `embeddings/` contains predicted CLIP image/text embeddings and trial indices
- `results.json` aggregates image, text, and prediction metrics

### Running individual stages

Train models only:

```bash
RUN_ID=exp_001 bash pipelines/train_models.sh
```

Generate predictions from existing models:

```bash
RUN_ID=exp_001 python3 scripts/predict_ridge_fmri_to_clip_multimodal.py
```

The `RUN_ID` environment variable ensures all components of an experiment share the same output directory.

## Notes

- `data/`, intermediate outputs, and run artifacts are not committed to version control
- Experiments are reproducible from code using the same `RUN_ID`
- Run directories in `results/` are treated as experiment outputs
- Main entrypoints are in `pipelines/`
