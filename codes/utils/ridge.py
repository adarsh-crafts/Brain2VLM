import argparse, os
import numpy as np
import joblib
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import psutil, os
def mem():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        type=str,
        default='',
        help="Target variable",
    )
    parser.add_argument(
        "--roi",
        required=True,
        type=str,
        nargs="*",
        help="use roi name",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )

    opt = parser.parse_args()
    target = opt.target
    roi = opt.roi

    backend = set_backend("numpy", on_error="warn")
    subject=opt.subject

    if target == 'c' or target == 'init_latent': # CVPR
        alpha = [0.000001,0.00001,0.0001,0.001,0.01, 0.1, 1]
    else: # text / GAN / depth decoding (with much larger number of voxels)
        alpha = [10000, 20000, 40000]

    ridge = RidgeCV(alphas=alpha, solver_params={"n_targets_batch": 1000})

    preprocess_pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
    )
    pipeline = make_pipeline(
        preprocess_pipeline,
        ridge,
    )    
    mridir = f'../../mrifeat/{subject}/'
    featdir = '../../nsdfeat/subjfeat/'
    savedir = f'../..//decoded/{subject}/'
    os.makedirs(savedir, exist_ok=True)

    # Collect ROI arrays then hstack once, freeing the list immediately
    X_list = []
    X_te_list = []
    for croi in roi:
        if 'conv' in target:
            arr = np.load(f'{mridir}/{subject}_{croi}_betas_ave_tr.npy', mmap_mode='r')
        else:
            arr = np.load(f'{mridir}/{subject}_{croi}_betas_tr.npy', mmap_mode='r')
        X_list.append(arr.astype("float32"))

        arr_te = np.load(f'{mridir}/{subject}_{croi}_betas_ave_te.npy', mmap_mode='r')
        X_te_list.append(arr_te.astype("float32"))

        # ---- Memory-safe stacking (avoid np.hstack peak copy) ----
        total_dim = sum(arr.shape[1] for arr in X_list)
        n_samples = X_list[0].shape[0]

        X = np.empty((n_samples, total_dim), dtype=np.float32)
        start = 0
        for arr in X_list:
            end = start + arr.shape[1]
            X[:, start:end] = arr
            start = end
        del X_list

        total_dim_te = sum(arr.shape[1] for arr in X_te_list)
        n_samples_te = X_te_list[0].shape[0]

        X_te = np.empty((n_samples_te, total_dim_te), dtype=np.float32)
        start = 0
        for arr in X_te_list:
            end = start + arr.shape[1]
            X_te[:, start:end] = arr
            start = end
        del X_te_list
    print(f'[MEM] after X/X_te load: {mem():.2f} GB')

    # Load Y via mmap (NO scaling, NO copy explosion)
    Y_raw = np.load(
        f'{featdir}/{subject}_each_{target}_tr.npy',
        mmap_mode='r'
    )
    Y = Y_raw.reshape([X.shape[0], -1])

    print(f'[MEM] after Y/Y_te load: {mem():.2f} GB')
    print(f'[MEM] available RAM: {psutil.virtual_memory().available / 1024**3:.2f} GB')

    print(f'Now making decoding model for... {subject}:  {roi}, {target}')
    print(f'X {X.shape}, Y {Y.shape}, X_te {X_te.shape}')
    print(f'[MEM] before fit: {mem():.2f} GB')
    print(ridge.get_params())

    # Train Ridge
    pipeline.fit(X, Y)
    print(f'[MEM] after fit: {mem():.2f} GB')

    # Free training data before predict allocates workspace
    del X
    del Y
    import gc
    gc.collect()

    print(f'[MEM] after del X,Y: {mem():.2f} GB')

    Y_te_raw = np.load(f'{featdir}/{subject}_ave_{target}_te.npy', mmap_mode='r')
    Y_te = Y_te_raw.reshape([X_te.shape[0], -1]).astype("float32")
    del Y_te_raw

    # Predictions
    scores = pipeline.predict(X_te)
    print(f'[MEM] after predict: {mem():.2f} GB')
    del X_te

    rs = correlation_score(Y_te.T, scores.T)
    print(f'Prediction accuracy is: {np.mean(rs):3.3}')

    np.save(f'{savedir}/{subject}_{"_".join(roi)}_scores_{target}.npy', scores)

if __name__ == "__main__":
    main()