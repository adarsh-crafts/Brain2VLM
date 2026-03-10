import h5py
from PIL import Image
import scipy.io
import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from einops import rearrange
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
import sys
sys.path.append("../utils/")
from nsd_access.nsda import NSDAccess
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def load_model_from_config(config, ckpt, gpu, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda(f"cuda:{gpu}")
    model.eval()
    return model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--imgidx",   required=True,  type=int)
    parser.add_argument("--gpu",      required=True,  type=int)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--subject",  required=True,  type=str)
    parser.add_argument("--method",   required=True,  type=str,
                        help="cvpr or text or gan or mlp")
    parser.add_argument("--t_noise",  type=int, default=500,
                        help="Noise timestep for q(x_t|x0), range [0, 1000]")

    opt = parser.parse_args()
    seed_everything(opt.seed)
    imgidx  = opt.imgidx
    gpu     = opt.gpu
    method  = opt.method
    subject = opt.subject
    t_noise = opt.t_noise

    precision       = 'autocast'
    precision_scope = autocast if precision == "autocast" else nullcontext

    # ------------------------------------------------------------------ #
    # Load NSD info                                                        #
    # ------------------------------------------------------------------ #
    nsd_expdesign = scipy.io.loadmat(
        '../../nsd/nsddata/experiments/nsd/nsd_expdesign.mat')
    sharedix = nsd_expdesign['sharedix'] - 1   # 1-based → 0-based

    nsda      = NSDAccess('../../nsd/')
    sf        = h5py.File(nsda.stimuli_file, 'r')
    sdataset  = sf.get('imgBrick')

    stims_ave = np.load(f'../../mrifeat/{subject}/{subject}_stims_ave.npy')

    tr_idx = np.zeros_like(stims_ave)
    for idx, s in enumerate(stims_ave):
        tr_idx[idx] = 0 if s in sharedix else 1

    # ------------------------------------------------------------------ #
    # Load Stable Diffusion (VAE + U-Net)                                 #
    # ------------------------------------------------------------------ #
    config = OmegaConf.load(
        './stable-diffusion/configs/stable-diffusion/v1-inference.yaml')
    ckpt   = './stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt'
    torch.cuda.set_device(gpu)
    model  = load_model_from_config(config, ckpt, gpu)

    device = (torch.device(f"cuda:{gpu}")
              if torch.cuda.is_available() else torch.device("cpu"))
    model  = model.to(device)

    # ------------------------------------------------------------------ #
    # Output dirs                                                          #
    # ------------------------------------------------------------------ #
    outdir      = f'../../decoded/image-{method}-coarse/{subject}/'
    sample_path = os.path.join(outdir, "samples")
    os.makedirs(sample_path, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Resolve test-image index                                             #
    # ------------------------------------------------------------------ #
    imgidx_te = np.where(tr_idx == 0)[0][imgidx]
    idx73k    = stims_ave[imgidx_te]
    print("RECON idx73k:", idx73k)

    # Save original stimulus
    Image.fromarray(
        np.squeeze(sdataset[idx73k, :, :, :]).astype(np.uint8)
    ).save(os.path.join(sample_path, f"{imgidx:05}_org.png"))

    # ------------------------------------------------------------------ #
    # Build predicted latent ẑ  (4 × 40 × 40)                            #
    # ------------------------------------------------------------------ #
    if method in ['cvpr', 'mlp', 'text']:
        roi_latent = 'early'
        if method == 'mlp':
            scores_latent = np.load(
                f'../../decoded/{subject}/{subject}_{roi_latent}_scores_init_latent_mlp.npy')
        else:
            scores_latent = np.load(
                f'../../decoded/{subject}/{subject}_{roi_latent}_scores_init_latent.npy')
        latent = scores_latent[imgidx, :]
        latent = (latent - latent.mean()) / (latent.std() + 1e-6)
        imgarr = torch.tensor(
            latent.reshape(4, 40, 40),
            device=device, dtype=model.dtype
        ).unsqueeze(0)                          # (1, 4, 40, 40)

    elif method == 'gan':
        from PIL import Image as PILImage
        gandir  = f'../../decoded/gan_recon_img/all_layers/{subject}/streams/'
        ganpath = (f'{gandir}/recon_image_normalized-VGG19-fc8-'
                   f'{subject}-streams-{imgidx:06}.tiff')
        im      = PILImage.open(ganpath).resize((512, 512))
        im_arr  = np.array(im).astype(np.float32) / 255.0
        im_arr  = im_arr[None].transpose(0, 3, 1, 2)
        im_tensor = torch.from_numpy(im_arr)
        im_tensor = (2. * im_tensor - 1.).to(device=device, dtype=model.dtype)
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    imgarr = model.get_first_stage_encoding(
                        model.encode_first_stage(im_tensor))
    else:
        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------ #
    # Step 1 — Coarse reconstruction  ẑ → x₀                             #
    #   Decode the predicted latent through the VAE decoder.              #
    #   No diffusion, no conditioning.                                    #
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                x0_pixel = model.decode_first_stage(imgarr)          # (1,3,H,W) in [-1,1]
                x0_vis   = torch.clamp((x0_pixel + 1.0) / 2.0,
                                       min=0.0, max=1.0)

                x0_np = 255. * rearrange(
                    x0_vis[0].cpu().numpy(), 'c h w -> h w c')

    coarse_path = os.path.join(sample_path, f"{imgidx:05}_coarse.png")
    Image.fromarray(x0_np.astype(np.uint8)).resize((512, 512)).save(coarse_path)
    print(f"Saved coarse reconstruction → {coarse_path}")

    # ------------------------------------------------------------------ #
    # Step 2 — Forward diffusion  x₀ → x_t  via  q(x_t | x₀)            #
    #   Re-encode x₀ back to latent space, then corrupt with noise at     #
    #   timestep t using the scheduler's closed-form formula:             #
    #       x_t = sqrt(ā_t) · x₀ + sqrt(1 - ā_t) · ε,  ε ~ N(0, I)     #
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():

                # Normalise x₀ back to [-1, 1] for the encoder
                x0_normed = (2. * x0_vis - 1.).to(device=device,
                                                   dtype=model.dtype)

                # Encode pixel x₀ → latent x₀_latent
                x0_latent = model.get_first_stage_encoding(
                    model.encode_first_stage(x0_normed))  # (1, 4, 40, 40)

                # Sample noise and apply q(x_t | x₀) at the chosen timestep
                t_tensor = torch.tensor(
                    [t_noise], device=device, dtype=torch.long)
                noise = torch.randn_like(x0_latent)

                # model.q_sample uses the pre-computed sqrt(ā_t) schedule
                x_t = model.q_sample(
                    x_start=x0_latent, t=t_tensor, noise=noise)  # (1, 4, 40, 40)

                # ---- optional: visualise x_t by decoding back to pixels ----
                x_t_decoded = model.decode_first_stage(x_t)
                x_t_vis     = torch.clamp((x_t_decoded + 1.0) / 2.0,
                                          min=0.0, max=1.0)
                x_t_np      = 255. * rearrange(
                    x_t_vis[0].cpu().numpy(), 'c h w -> h w c')

    xt_path = os.path.join(sample_path, f"{imgidx:05}_xt_t{t_noise}.png")
    Image.fromarray(x_t_np.astype(np.uint8)).resize((512, 512)).save(xt_path)
    print(f"Saved x_t (t={t_noise}) visualisation → {xt_path}")

    # ------------------------------------------------------------------ #
    # Also save the raw x_t latent tensor for downstream U-Net use        #
    # ------------------------------------------------------------------ #
    xt_latent_path = os.path.join(
        sample_path, f"{imgidx:05}_xt_latent_t{t_noise}.pt")
    torch.save(x_t.cpu(), xt_latent_path)
    print(f"Saved x_t latent tensor → {xt_latent_path}")

    print("\nDone. Outputs:")
    print(f"  Original  : {imgidx:05}_org.png")
    print(f"  Coarse x₀ : {imgidx:05}_coarse.png")
    print(f"  x_t visual : {imgidx:05}_xt_t{t_noise}.png")
    print(f"  x_t tensor : {imgidx:05}_xt_latent_t{t_noise}.pt")


if __name__ == "__main__":
    main()