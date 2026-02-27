
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

cd codes/diffusion_sd1/
python diffusion_decoding_copy.py --imgidx 0 --gpu 0 --subject subj01 --method cvpr
```

# Evaluation
```
cd codes/utils/
python img2feat_decoded_copy.py --gpu 0 --subject subj01 --method cvpr
python identification.py --usefeat inception --subject subj01 --method cvpr
```

# Acknowledgement
Our codebase builds on these repositories. We would like to thank the authors. 

> https://github.com/yu-takagi/StableDiffusionReconstruction

> https://github.com/CompVis/stable-diffusion

> https://github.com/gallantlab/himalaya

> https://github.com/tknapen/nsd_access

> https://github.com/KamitaniLab/brain-decoding-cookbook-public
