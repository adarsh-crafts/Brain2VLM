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
```bash
bash train_linear_decoders.sh
```
Outputs:

- metrics → `results/linear_decoders/`
- models → `models/` (not tracked)

Notes
Trained models and raw data are not committed.

Results JSON files contain evaluation metrics.

Main entrypoints are in `pipelines/`.
