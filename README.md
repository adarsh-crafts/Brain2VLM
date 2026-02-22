abstract here.

# Environment setup
1. Download ``nsddata``, ``nsddata_betas``, and ``nsddata_stimuli`` from NSD and place them under the ``nsd`` directory.
2. ``pip install -r requirements.txt``
3. Install Stable Diffusion v1.4 (under the ``diffusion_sd1/`` directory), download checkpoint (``sd-v1-4.ckpt``), and place it under the ``codes/diffusion_sd1/stable-diffusion/models/ldm/stable-diffusion-v1/`` directory.
4. For incorporating GAN, install ``bdpy`` (under the ``gan/`` directory), download ``VGG_ILSVRC_19_layers`` and ``bvlc_reference_caffenet_generator_ILSVRC2012_Training`` from https://figshare.com/articles/dataset/brain-decoding-cookbook/21564384, and place them under the ``codes/gan/models/pytorch/`` directory.
5. For incorporating decoded depth, install Stable Diffusion v2.0 (under the ``diffusion_sd2/`` directory), download checkpoint (``512-depth-ema.ckpt``), and place it under the ``codes/diffusion_sd2/stablediffusion/models/`` directory. 

# MRI Preprocessing
```
cd codes/utils/
python make_subjmri.py --subject subj01
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

# Reconstruction with Decoded Text Prompt
```
cd codes/caption/BLIP/
python img2feat_blip.py --gpu 0

cd codes/utils/
python make_subjstim.py --featname blip --use_stim ave --subject subj01
python make_subjstim.py --featname blip --use_stim each --subject subj01
python ridge.py --target blip --roi early ventral midventral midlateral lateral parietal  --subject subj01

cd codes/caption/BLIP/
python decode_captions.py --subject subj01

cd codes/diffusion_sd1/
python diffusion_decoding.py --imgidx 0 --gpu 1 --subject subj01 --method text
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

> https://github.com/KamitaniLab/bdpy

> https://github.com/KamitaniLab/brain-decoding-cookbook-public