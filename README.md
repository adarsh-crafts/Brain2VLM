[![Python](https://img.shields.io/badge/Python-3.11.14-blue)](https://www.python.org/)

# Startup
1. Setup the env and data files.  
(use [steps-to-setup-ec2.md](steps-to-setup-ec2.md))
3. Install Stable Diffusion v1.4 (under the ``diffusion_sd1/`` directory), download checkpoint (``sd-v1-4.ckpt``), and place it under the ``codes/diffusion_sd1/stable-diffusion/models/ldm/stable-diffusion-v1/`` directory.  
(use [notebook](my-code/download_weights.ipynb).)

# MRI Preprocessing
```
cd codes/utils/
python make_subjmri_copy.py --subject subj01
```

# Reconstruction based on CVPR method
```
cd codes/utils/
python img2feat_sd_copy_batching.py  --imgidx 0 73000 --gpu 0 --batch_size 30
python make_subjstim_copy.py --featname init_latent --use_stim each --subject subj01
python make_subjstim_copy.py --featname init_latent --use_stim ave --subject subj01
python make_subjstim_copy.py --featname c --use_stim each --subject subj01
python make_subjstim_copy.py --featname c --use_stim ave --subject subj01

python ridge_copy.py --target c --roi ventral --subject subj01
python ridge_copy.py --target init_latent --roi early --subject subj01

python -u mlp.py --target c --roi ventral --subject subj01 2>&1 | tee logs/run_normal_mlp_c.txt
python -u mlp.py --target init_latent --roi early --subject subj01 2>&1 | tee logs/run_normal_mlp_c.tx

cd codes/diffusion_sd1/
python diffusion_decoding_copy.py --imgidx 0 --gpu 0 --subject subj01 --method cvpr/mlp
```
or
```
for i in {0..981}; do
    echo "Running imgidx $i"
    python diffusion_decoding_copy.py --imgidx $i --gpu 0 --subject subj01 --method cvpr/mlp
done 2>&1 | tee logs/run_recon_cvpr/mlp.txt
```

## Best MLP Config:
```

```

# Evaluation
```
cd codes/utils/
python img2feat_decoded_copy.py --gpu 0 --subject subj01 --method cvpr
python img2feat_decoded_copy.py --gpu 0 --subject subj01 --method mlp

python identification.py --usefeat inception --subject subj01 --method cvpr
python identification.py --usefeat inception --subject subj01 --method mlp
```

To calculate all scores in one go:
```
python aggr_identification_scores
```

```
python retrieval.py --pred_file ../../decoded/subj01/subj01_ventral_scores_init_latent.npy
python retrieval.py --pred_file ../../decoded/subj01/subj01_early_scores_init_latent_mlp.npy
```

# Acknowledgement
Our codebase builds on these repositories. We would like to thank the authors. 

> https://github.com/yu-takagi/StableDiffusionReconstruction

> https://github.com/CompVis/stable-diffusion

> https://github.com/openai/CLIP

> https://github.com/CompVis/taming-transformers

> https://github.com/tknapen/nsd_access

> https://github.com/KamitaniLab/brain-decoding-cookbook-public

---

# Ablation Studies

## 1. init_latent (early cortex)
### Depth Ablation (Nonlinearity Study)
```
{
DEPTHS=(0 1 2 4 6)

for d in "${DEPTHS[@]}"
do
for s in 0 1 2
do
echo "DEPTH=$d SEED=$s"

python mlp_ablations.py \
--subject subj01 \
--roi early \
--target init_latent \
--depth $d \
--hidden_dim 2048 \
--data_frac 1.0 \
--seed $s

done
done
} 2>&1 | tee logs/run_latent_depth_ablation.txt
```

### Width Ablation (Capacity Study)
```
{
WIDTHS=(512 1024 2048 4096)

for w in "${WIDTHS[@]}"
do
for s in 0 1 2
do
echo "WIDTH=$w SEED=$s"

python mlp_ablations.py \
--subject subj01 \
--roi early \
--target init_latent \
--depth 2 \
--hidden_dim $w \
--data_frac 1.0 \
--seed $s

done
done
} 2>&1 | tee logs/run_latent_width_ablation.txt

```

### Data Efficiency Study
```
{
FRACS=(0.1 0.25 0.5 0.75 1.0)

for f in "${FRACS[@]}"
do
for s in 0 1 2
do
echo "FRAC=$f SEED=$s"

python mlp_ablations.py \
--subject subj01 \
--roi early \
--target init_latent \
--depth 2 \
--hidden_dim 2048 \
--data_frac $f \
--seed $s

done
done
} 2>&1 | tee logs/run_latent_frac_ablation.txt
```

## 2. c (ventral cortex)
### Depth Ablation (Nonlinearity Study)
```
{
DEPTHS=(0 1 2 4 6)

for d in "${DEPTHS[@]}"
do
for s in 0 1 2
do
echo "DEPTH=$d SEED=$s"

python mlp_ablations.py \
--subject subj01 \
--roi ventral \
--target c \
--depth $d \
--hidden_dim 2048 \
--data_frac 1.0 \
--seed $s

done
done
} 2>&1 | tee logs/run_c_depth_ablation.txt
```

### Width Ablation (Capacity Study)
```
{
WIDTHS=(512 1024 2048 4096)

for w in "${WIDTHS[@]}"
do
for s in 0 1 2
do
echo "WIDTH=$w SEED=$s"

python mlp_ablations.py \
--subject subj01 \
--roi ventral \
--target c \
--depth 2 \
--hidden_dim $w \
--data_frac 1.0 \
--seed $s

done
done
} 2>&1 | tee logs/run_c_width_ablation.txt

```

### Data Efficiency Study
```
{
FRACS=(0.1 0.25 0.5 0.75 1.0)

for f in "${FRACS[@]}"
do
for s in 0 1 2
do
echo "FRAC=$f SEED=$s"

python mlp_ablations.py \
--subject subj01 \
--roi ventral \
--target c \
--depth 2 \
--hidden_dim 2048 \
--data_frac $f \
--seed $s

done
done
} 2>&1 | tee logs/run_c_frac_ablation.txt
```