import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import os


# -------------------------------------------------
# Load original features
# -------------------------------------------------

def load_org_matrix(nimage, subject, method, usefeat):

    featdir = f'../../identification/{method}/{subject}'
    feats = []

    for imgid in range(nimage):
        f = np.load(f"{featdir}/{imgid:05}_org_{usefeat}.npy").flatten()
        feats.append(f)

    return np.stack(feats)


# -------------------------------------------------
# Load generated features
# -------------------------------------------------

def load_gen_matrix(nimage, subject, method, usefeat):

    featdir = f'../../identification/{method}/{subject}'
    nrep = 5

    feats = []

    for imgid in range(nimage):

        reps = []

        for rep in range(nrep):
            f = np.load(f"{featdir}/{imgid:05}_{rep:03}_{usefeat}.npy").flatten()
            reps.append(f)

        reps = np.mean(np.stack(reps), axis=0)
        feats.append(reps)

    return np.stack(feats)


# -------------------------------------------------
# Fast correlation matrix
# -------------------------------------------------

def corr_matrix(A, B):

    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)

    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)

    return A @ B.T


# -------------------------------------------------
# Identification accuracy
# -------------------------------------------------

def identification_accuracy(R):

    n = R.shape[0]

    correct = 0
    total = n * (n - 1)

    diag = np.diag(R)

    for i in range(n):

        r_true = diag[i]
        r_fake = np.delete(R[i], i)

        correct += np.sum(r_true > r_fake)

    return correct / total


# -------------------------------------------------
# Detect feature names automatically
# -------------------------------------------------

def detect_features(method, subject):

    featdir = f'../../identification/{method}/{subject}'

    features = sorted({
        f.split("_org_")[1].replace(".npy","")
        for f in os.listdir(featdir)
        if "_org_" in f
    })

    return features


# -------------------------------------------------
# Infer model + layer metadata
# -------------------------------------------------

def infer_model_layer(featname):

    model = "Unknown"
    layer = -1

    # CLIP
    if featname.startswith("clip_h"):
        model = "CLIP"
        layer = int(featname.split("h")[1])

    elif featname == "clip":
        model = "CLIP"
        layer = 25

    # DINO
    elif featname.startswith("dino_h"):
        model = "DINOv2"
        layer = int(featname.split("h")[1])

    elif featname == "dino":
        model = "DINOv2"
        layer = 24

    # ResNet
    elif featname.startswith("resnet"):
        model = "ResNet"

        if "conv1" in featname:
            layer = 0
        elif "l1" in featname:
            layer = 1
        elif "l2" in featname:
            layer = 2
        elif "l3" in featname:
            layer = 3
        elif "l4" in featname:
            layer = 4
        elif "avgpool" in featname:
            layer = 5

    # AlexNet
    elif featname.startswith("alex"):
        model = "AlexNet"

        if "conv1" in featname:
            layer = 1
        elif "conv2" in featname:
            layer = 2
        elif "conv3" in featname:
            layer = 3
        elif "conv4" in featname:
            layer = 4
        elif "conv5" in featname:
            layer = 5
        elif "fc6" in featname:
            layer = 6
        elif "fc7" in featname:
            layer = 7
        elif "fc8" in featname:
            layer = 8

    # Inception
    elif featname.startswith("inception"):
        model = "Inception"

        if "mix5" in featname:
            layer = 5
        elif "mix6" in featname:
            layer = 6
        elif "mix7" in featname:
            layer = 7
        elif "avgpool" in featname:
            layer = 8

    return model, layer


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--subject", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--out_csv", default="identification_results.csv")

    opt = parser.parse_args()

    subject = opt.subject
    methods = opt.methods
    nimage = 982

    results = []

    for method in methods:

        print(f"\nEvaluating method: {method}")

        features = detect_features(method, subject)

        for featname in tqdm(features):

            try:

                org = load_org_matrix(nimage, subject, method, featname)
                gen = load_gen_matrix(nimage, subject, method, featname)

            except FileNotFoundError:
                continue

            R = corr_matrix(org, gen)

            acc = identification_accuracy(R)

            model, layer = infer_model_layer(featname)

            results.append({
                "method": method,
                "model": model,
                "layer": layer,
                "feature": featname,
                "accuracy": acc
            })

    df = pd.DataFrame(results)

    df.to_csv(opt.out_csv, index=False)

    print("\nSaved results to:", opt.out_csv)


if __name__ == "__main__":
    main()