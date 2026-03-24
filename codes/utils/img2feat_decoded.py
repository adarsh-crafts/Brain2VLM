import argparse
import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import (
    ResNet50_Weights,
    AlexNet_Weights,
    Inception_V3_Weights
)

from transformers import CLIPProcessor, CLIPModel
import timm


def save_feat(path, feat):
    np.save(path, feat.astype(np.float32))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--method", required=True)

    opt = parser.parse_args()

    torch.set_grad_enabled(False)

    device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(opt.gpu)

    imglist = sorted(
        glob.glob(f'../../decoded/image-{opt.method}/{opt.subject}/samples/*')
    )

    outdir = f'../../identification/{opt.method}/{opt.subject}/'
    os.makedirs(outdir, exist_ok=True)

    print("Loading models...")

    # -------------------------------------------------
    # Image preprocessing
    # -------------------------------------------------

    preprocess_224 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    preprocess_299 = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    preprocess_518 = transforms.Compose([
        transforms.Resize(518),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # -------------------------------------------------
    # ResNet50 (multiple internal layers)
    # -------------------------------------------------

    resnet = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet.eval().to(device)

    resnet = create_feature_extractor(
        resnet,
        return_nodes={
            "conv1":"conv1",
            "layer1.0.relu":"l1_0",
            "layer1.1.relu":"l1_1",
            "layer2.0.relu":"l2_0",
            "layer2.1.relu":"l2_1",
            "layer3.0.relu":"l3_0",
            "layer3.1.relu":"l3_1",
            "layer4.0.relu":"l4_0",
            "layer4.1.relu":"l4_1",
            "avgpool":"avgpool"
        }
    )

    # -------------------------------------------------
    # AlexNet (all conv layers)
    # -------------------------------------------------

    alexnet = torchvision.models.alexnet(weights=AlexNet_Weights.DEFAULT)
    alexnet.eval().to(device)

    alexnet = create_feature_extractor(
        alexnet,
        return_nodes={
            "features.0":"conv1",
            "features.3":"conv2",
            "features.6":"conv3",
            "features.8":"conv4",
            "features.10":"conv5",
            "classifier.1":"fc6",
            "classifier.4":"fc7",
            "classifier.6":"fc8"
        }
    )

    # -------------------------------------------------
    # InceptionV3
    # -------------------------------------------------

    inception = torchvision.models.inception_v3(
        weights=Inception_V3_Weights.DEFAULT
    )

    inception.eval().to(device)

    inception = create_feature_extractor(
        inception,
        return_nodes={
            "Mixed_5b":"mix5b",
            "Mixed_5c":"mix5c",
            "Mixed_5d":"mix5d",
            "Mixed_6a":"mix6a",
            "Mixed_6b":"mix6b",
            "Mixed_6c":"mix6c",
            "Mixed_6d":"mix6d",
            "Mixed_6e":"mix6e",
            "Mixed_7a":"mix7a",
            "Mixed_7b":"mix7b",
            "Mixed_7c":"mix7c",
            "avgpool":"avgpool"
        }
    )

    # -------------------------------------------------
    # CLIP
    # -------------------------------------------------

    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14"
    ).to(device)

    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14"
    )

    # -------------------------------------------------
    # DINOv2
    # -------------------------------------------------

    dino = timm.create_model(
        "vit_large_patch14_dinov2",
        pretrained=True
    )

    dino.eval().to(device)

    print("Starting feature extraction")

    for img_path in tqdm(imglist):

        imgname = os.path.basename(img_path).split(".")[0]
        image = Image.open(img_path).convert("RGB")
        fname = f"{outdir}/{imgname}"

        # -------------------------------------------------
        # ResNet
        # -------------------------------------------------

        inp = preprocess_224(image).unsqueeze(0).to(device)
        feat = resnet(inp)

        for k,v in feat.items():
            save_feat(f"{fname}_resnet_{k}.npy", v.flatten().cpu().numpy())

        # -------------------------------------------------
        # AlexNet
        # -------------------------------------------------

        feat = alexnet(inp)

        for k,v in feat.items():
            save_feat(f"{fname}_alex_{k}.npy", v.flatten().cpu().numpy())

        # -------------------------------------------------
        # Inception
        # -------------------------------------------------

        inp299 = preprocess_299(image).unsqueeze(0).to(device)
        feat = inception(inp299)

        for k,v in feat.items():
            save_feat(f"{fname}_inception_{k}.npy", v.flatten().cpu().numpy())

        # -------------------------------------------------
        # CLIP layers
        # -------------------------------------------------

        clip_inputs = clip_processor(
            images=image,
            return_tensors="pt"
        ).to(device)

        clip_out = clip_model.vision_model(
            pixel_values=clip_inputs["pixel_values"],
            output_hidden_states=True
        )

        clip_embed = clip_out.pooler_output.squeeze().cpu().numpy()
        save_feat(f"{fname}_clip.npy", clip_embed)

        hidden = clip_out.hidden_states

        for i,h in enumerate(hidden):
            save_feat(
                f"{fname}_clip_h{i}.npy",
                h.flatten().cpu().numpy()
            )

        # -------------------------------------------------
        # DINO layers
        # -------------------------------------------------

        dino_inp = preprocess_518(image).unsqueeze(0).to(device)

        with torch.no_grad():

            x = dino.patch_embed(dino_inp)
            cls_token = dino.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + dino.pos_embed

            for i,block in enumerate(dino.blocks):
                x = block(x)
                save_feat(
                    f"{fname}_dino_h{i}.npy",
                    x.flatten().cpu().numpy()
                )

    print("Feature extraction complete")


if __name__ == "__main__":
    main()