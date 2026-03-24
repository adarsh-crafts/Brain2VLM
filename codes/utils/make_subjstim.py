import numpy as np
import scipy.io
from tqdm import tqdm
import argparse
import os


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--featname",
        type=str,
        default='',
        help="Target variable",
    )
    parser.add_argument(
        "--use_stim",
        type=str,
        default='',
        help="ave or each",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="subject name: subj01 or subj02 or subj05 or subj07",
    )

    opt = parser.parse_args()

    subject = opt.subject
    use_stim = opt.use_stim
    featname = opt.featname

    topdir = '../../nsdfeat/'
    savedir = f'{topdir}/subjfeat/'
    featdir = f'{topdir}/{featname}/'

    os.makedirs(savedir, exist_ok=True)

    # Load experiment design
    nsd_expdesign = scipy.io.loadmat(
        '../../nsd/nsddata/experiments/nsd/nsd_expdesign.mat'
    )

    # Convert to 0-based indexing
    sharedix = nsd_expdesign['sharedix'].flatten() - 1
    sharedix_set = set(sharedix.tolist())  # O(1) membership

    # Load stimulus indices
    if use_stim == 'ave':
        stims = np.load(f'../../mrifeat/{subject}/{subject}_stims_ave.npy')
    else:
        stims = np.load(f'../../mrifeat/{subject}/{subject}_stims.npy')

    n_total = len(stims)

    # -----------------------------
    # Compute train/test split
    # -----------------------------
    tr_idx = np.zeros(n_total)

    for i, s in enumerate(stims):
        if s in sharedix_set:
            tr_idx[i] = 0
        else:
            tr_idx[i] = 1

    train_mask = tr_idx == 1
    test_mask = tr_idx == 0

    n_train = np.sum(train_mask)
    n_test = np.sum(test_mask)

    # -----------------------------
    # Determine feature dimensionality
    # -----------------------------
    first_feat = np.load(f'{featdir}/{stims[0]:06}.npy')
    feat_shape = first_feat.shape
    feat_dtype = first_feat.dtype

    # Ensure 2D stacking behavior identical to original np.stack
    feat_dim = first_feat.size

    # -----------------------------
    # Preallocate output arrays
    # -----------------------------
    feats_tr = np.empty((n_train, feat_dim), dtype=feat_dtype)
    feats_te = np.empty((n_test, feat_dim), dtype=feat_dtype)

    tr_counter = 0
    te_counter = 0

    # -----------------------------
    # Stream features directly
    # -----------------------------
    for idx, s in tqdm(enumerate(stims)):

        feat = np.load(f'{featdir}/{s:06}.npy')
        feat_flat = feat.reshape(-1)

        if train_mask[idx]:
            feats_tr[tr_counter] = feat_flat
            tr_counter += 1
        else:
            feats_te[te_counter] = feat_flat
            te_counter += 1

    # -----------------------------
    # Save outputs (IDENTICAL filenames)
    # -----------------------------
    np.save(f'../../mrifeat/{subject}/{subject}_stims_tridx.npy', tr_idx)

    np.save(f'{savedir}/{subject}_{use_stim}_{featname}_tr.npy', feats_tr)
    np.save(f'{savedir}/{subject}_{use_stim}_{featname}_te.npy', feats_te)


if __name__ == "__main__":
    main()
