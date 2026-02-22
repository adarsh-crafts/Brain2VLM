import argparse
import os
import numpy as np
import pandas as pd
from nsd_access import NSDAccess
import scipy.io
import gc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )

    opt = parser.parse_args()
    subject = opt.subject
    atlasname = 'streams'
    
    nsda = NSDAccess('../../nsd/')
    nsd_expdesign = scipy.io.loadmat('../../nsd/nsddata/experiments/nsd/nsd_expdesign.mat')

    # Note that most of nsd_expdesign indices are 1-base index!
    # This is why subtracting 1
    sharedix = nsd_expdesign['sharedix'] -1 

    atlas = nsda.read_atlas_results(subject=subject, atlas=atlasname, data_format='func1pt8mm')
    atlas_mask = atlas[0].transpose([2,1,0])

    beh_list = []
    for i in range(1,38):
        beh = nsda.read_behavior(subject=subject, 
                                session_index=i)
        beh_list.append(beh)
    behs = pd.concat(beh_list, ignore_index=True)

    # Caution: 73KID is 1-based! https://cvnlab.slite.page/p/fRv4lz5V2F/Behavioral-data
    stims_unique = behs['73KID'].unique() - 1
    stims_all = behs['73KID'] - 1

    savedir = f'../../mrifeat/{subject}/'
    os.makedirs(savedir, exist_ok=True)

    if not os.path.exists(f'{savedir}/{subject}_stims.npy'):
        np.save(f'{savedir}/{subject}_stims.npy',stims_all)
        np.save(f'{savedir}/{subject}_stims_ave.npy',stims_unique)

    for roi,val in atlas[1].items():
        print(roi,val)
        if val == 0:
            print('SKIP')
            continue
        else:
            betas_roi_list = []
            for i in range(1,38):
                print(i)
                beta_trial = nsda.read_betas(subject=subject, 
                                        session_index=i, 
                                        trial_index=[], # empty list as index means get all for this session
                                        data_type='betas_fithrf_GLMdenoise_RR',
                                        data_format='func1pt8mm')
                beta_roi = beta_trial[:, atlas_mask==val]
                betas_roi_list.append(beta_roi)
                del beta_trial
            betas_roi = np.concatenate(betas_roi_list, axis=0)
            del betas_roi_list

        print(betas_roi.shape)
        
        # Averaging for each stimulus
        betas_roi_ave = []
        for stim in stims_unique:
            stim_mean = np.mean(betas_roi[stims_all == stim,:],axis=0)
            betas_roi_ave.append(stim_mean)
        betas_roi_ave = np.stack(betas_roi_ave)
        print(betas_roi_ave.shape)
        
        # Train/Test Split
        # ALLDATA
        shared_mask = np.isin(stims_all, sharedix)
        betas_te = betas_roi[shared_mask]
        betas_tr = betas_roi[~shared_mask]
        
        # AVERAGED DATA        
        shared_mask_ave = np.isin(stims_unique, sharedix)
        betas_ave_te = betas_roi_ave[shared_mask_ave]
        betas_ave_tr = betas_roi_ave[~shared_mask_ave]    
        
        # Save
        np.save(f'{savedir}/{subject}_{roi}_betas_tr.npy',betas_tr)
        np.save(f'{savedir}/{subject}_{roi}_betas_te.npy',betas_te)
        np.save(f'{savedir}/{subject}_{roi}_betas_ave_tr.npy',betas_ave_tr)
        np.save(f'{savedir}/{subject}_{roi}_betas_ave_te.npy',betas_ave_te)

        del betas_roi
        gc.collect()


if __name__ == "__main__":
    main()
