import os
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import scipy.io


def load_gt_latents(folder):

    files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
    latents = []

    for f in files:
        path = os.path.join(folder, f)
        arr = np.load(path)

        if arr.ndim > 1:
            arr = arr.reshape(-1)

        latents.append(arr)

    latents = np.stack(latents)

    return latents


def load_pred_matrix(path):

    arr = np.load(path)

    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    return arr


def load_test_stim_ids(stim_file, expdesign_file):

    stims = np.load(stim_file)

    nsd_expdesign = scipy.io.loadmat(expdesign_file)
    sharedix = nsd_expdesign['sharedix'].flatten() - 1

    shared_set = set(sharedix)

    test_ids = [s for s in stims if s in shared_set]

    return np.array(test_ids)


def retrieval(pred_latents, gt_latents, stim_ids):

    N = pred_latents.shape[0]

    top1 = 0
    top5 = 0
    top10 = 0

    for i in tqdm(range(N)):

        z_pred = pred_latents[i].reshape(1, -1)

        sims = cosine_similarity(z_pred, gt_latents)[0]

        ranking = np.argsort(-sims)

        correct_id = stim_ids[i]

        if correct_id in ranking[:1]:
            top1 += 1

        if correct_id in ranking[:5]:
            top5 += 1

        if correct_id in ranking[:10]:
            top10 += 1

    results = {
        "Top1": top1 / N,
        "Top5": top5 / N,
        "Top10": top10 / N
    }

    return results


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred_file",
        required=True,
        help="predicted latent npy matrix"
    )

    parser.add_argument(
        "--gt_dir",
        default="../../nsdfeat/init_latent/",
        help="ground truth latent folder"
    )

    parser.add_argument(
        "--stim_file",
        default="../../mrifeat/subj01/subj01_stims.npy",
        help="stimulus id file"
    )

    parser.add_argument(
        "--expdesign",
        default="../../nsd/nsddata/experiments/nsd/nsd_expdesign.mat",
        help="nsd expdesign file"
    )

    args = parser.parse_args()

    print("\nLoading ground truth latents...")
    gt_latents = load_gt_latents(args.gt_dir)

    print("GT shape:", gt_latents.shape)

    print("\nLoading predicted latents...")
    pred_latents = load_pred_matrix(args.pred_file)

    print("Pred shape:", pred_latents.shape)

    print("\nLoading stimulus mapping...")
    test_stim_ids = load_test_stim_ids(args.stim_file, args.expdesign)

    print("Test stimuli:", len(test_stim_ids))

    if len(test_stim_ids) != pred_latents.shape[0]:

        raise ValueError(
            "Prediction count and test stimulus count do not match"
        )

    print("\nRunning retrieval evaluation...\n")

    results = retrieval(pred_latents, gt_latents, test_stim_ids)

    print("\nFinal Results")
    print("--------------------------")

    print(f"Queries : {pred_latents.shape[0]}")
    print(f"Gallery : {gt_latents.shape[0]}")

    print(f"\nTop-1  : {results['Top1']*100:.2f}")
    print(f"Top-5  : {results['Top5']*100:.2f}")
    print(f"Top-10 : {results['Top10']*100:.2f}")


if __name__ == "__main__":
    main()