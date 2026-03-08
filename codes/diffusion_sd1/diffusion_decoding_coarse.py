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

    parser.add_argument(
        "--imgidx",
        required=True,
        type=int,
        help="img idx"
    )
    parser.add_argument(
        "--gpu",
        required=True,
        type=int,
        help="gpu"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--subject",
        required=True,
        type=str,
        default=None,
        help="subject name: subj01 or subj02 or subj05 or subj07 for full-data subjects",
    )
    parser.add_argument(
        "--method",
        required=True,
        type=str,
        help="cvpr or text or gan or mlp",
    )

    # Set parameters
    opt = parser.parse_args()
    seed_everything(opt.seed)
    imgidx = opt.imgidx
    gpu = opt.gpu
    method = opt.method
    subject = opt.subject

    precision = 'autocast'
    precision_scope = autocast if precision == "autocast" else nullcontext

    # Load NSD information
    nsd_expdesign = scipy.io.loadmat('../../nsd/nsddata/experiments/nsd/nsd_expdesign.mat')

    # Note that most of them are 1-base index — subtract 1
    sharedix = nsd_expdesign['sharedix'] - 1

    nsda = NSDAccess('../../nsd/')
    sf = h5py.File(nsda.stimuli_file, 'r')
    sdataset = sf.get('imgBrick')

    stims_ave = np.load(f'../../mrifeat/{subject}/{subject}_stims_ave.npy')

    tr_idx = np.zeros_like(stims_ave)
    for idx, s in enumerate(stims_ave):
        if s in sharedix:
            tr_idx[idx] = 0
        else:
            tr_idx[idx] = 1

    # Load Stable Diffusion Model (VAE only needed, but we load the full model)
    config = './stable-diffusion/configs/stable-diffusion/v1-inference.yaml'
    ckpt = './stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt'
    config = OmegaConf.load(f"{config}")
    torch.cuda.set_device(gpu)
    model = load_model_from_config(config, f"{ckpt}", gpu)

    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # Output directories
    outdir = f'../../decoded/image-{method}-coarse/{subject}/'
    os.makedirs(outdir, exist_ok=True)
    sample_path = os.path.join(outdir, "samples")
    os.makedirs(sample_path, exist_ok=True)

    # Resolve test image index
    imgidx_te = np.where(tr_idx == 0)[0][imgidx]
    idx73k = stims_ave[imgidx_te]
    print("RECON idx73k:", idx73k)

    # Save original stimulus
    Image.fromarray(np.squeeze(sdataset[idx73k, :, :, :]).astype(np.uint8)).save(
        os.path.join(sample_path, f"{imgidx:05}_org.png"))

    # ------------------------------------------------------------------
    # Build imgarr: the predicted latent ẑ (4×40×40) from the chosen method
    # ------------------------------------------------------------------
    if method in ['cvpr', 'mlp', 'text']:
        roi_latent = 'early'
        if method == 'mlp':
            scores_latent = np.load(f'../../decoded/{subject}/{subject}_{roi_latent}_scores_init_latent_mlp.npy')
        else:
            scores_latent = np.load(f'../../decoded/{subject}/{subject}_{roi_latent}_scores_init_latent.npy')
        latent = scores_latent[imgidx, :]
        latent = (latent - latent.mean()) / (latent.std() + 1e-6)

        imgarr = torch.tensor(
            latent.reshape(4, 40, 40),
            device=device,
            dtype=model.dtype
        ).unsqueeze(0)  # shape: (1, 4, 40, 40)

    elif method == 'gan':
        # GAN reconstructions live in pixel space — encode to latent first
        from PIL import Image as PILImage
        import PIL
        gandir = f'../../decoded/gan_recon_img/all_layers/{subject}/streams/'
        ganpath = f'{gandir}/recon_image_normalized-VGG19-fc8-{subject}-streams-{imgidx:06}.tiff'
        im = PILImage.open(ganpath).resize((512, 512))
        im_arr = np.array(im).astype(np.float32) / 255.0
        im_arr = im_arr[None].transpose(0, 3, 1, 2)
        im_tensor = torch.from_numpy(im_arr)
        im_tensor = (2. * im_tensor - 1.).to(device=device, dtype=model.dtype)
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    imgarr = model.get_first_stage_encoding(model.encode_first_stage(im_tensor))
    else:
        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # Coarse reconstruction: decode ẑ directly through the VAE — no diffusion,
    # no conditioning on ĉ. Corresponds to the "Coarse Reconstruction" step
    # in panel (c) of the Brain2VLM figure.
    # ------------------------------------------------------------------
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                x_samples = model.decode_first_stage(imgarr)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                for x_sample in x_samples:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')

    out_path = os.path.join(sample_path, f"{imgidx:05}_coarse.png")
    Image.fromarray(x_sample.astype(np.uint8)).resize((512, 512)).save(out_path)
    print(f"Saved coarse reconstruction to {out_path}")


if __name__ == "__main__":
    main()