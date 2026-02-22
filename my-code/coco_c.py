import numpy as np
import scipy.io
import pandas as pd
import argparse
from pycocotools.coco import COCO

parser = argparse.ArgumentParser()
parser.add_argument("--imgidx", required=True, type=int, nargs="+", help="image index or range (start end)")
parser.add_argument("--subject", required=True, type=str)
opt = parser.parse_args()

subject = opt.subject
imgidx_range = range(opt.imgidx[0], opt.imgidx[1]) if len(opt.imgidx) == 2 else [opt.imgidx[0]]

# ===============================
# 1) Reproduce diffusion indexing
# ===============================
nsd_expdesign = scipy.io.loadmat('../nsd/nsddata/experiments/nsd/nsd_expdesign.mat')
sharedix = nsd_expdesign['sharedix'].squeeze() - 1

stims_ave = np.load(f'../mrifeat/{subject}/{subject}_stims_ave.npy')

tr_idx = np.zeros_like(stims_ave)
for idx, s in enumerate(stims_ave):
    tr_idx[idx] = 0 if s in sharedix else 1

test_indices = np.where(tr_idx == 0)[0]

# ===============================
# 2) Load stim info + COCO files once
# ===============================
stim_info = pd.read_csv('../nsd/nsddata/experiments/nsd/nsd_stim_info_merged.csv')

coco_train = COCO('../nsd/nsddata_stimuli/stimuli/nsd/annotations/captions_train2017.json')
coco_val   = COCO('../nsd/nsddata_stimuli/stimuli/nsd/annotations/captions_val2017.json')

# For category labels
inst_train = COCO('../nsd/nsddata_stimuli/stimuli/nsd/annotations/instances_train2017.json')
inst_val   = COCO('../nsd/nsddata_stimuli/stimuli/nsd/annotations/instances_val2017.json')

for imgidx in imgidx_range:
    imgidx_te = test_indices[imgidx]
    idx73k = int(stims_ave[imgidx_te])

    row = stim_info.iloc[idx73k]

    coco_id    = int(row['cocoId'])
    coco_split = row['cocoSplit']

    coco_cap  = coco_train if coco_split == 'train2017' else coco_val
    coco_inst = inst_train  if coco_split == 'train2017' else inst_val

    # Captions
    ann_ids = coco_cap.getAnnIds(imgIds=[coco_id])
    anns    = coco_cap.loadAnns(ann_ids)

    # Categories
    inst_ann_ids = coco_inst.getAnnIds(imgIds=[coco_id])
    inst_anns    = coco_inst.loadAnns(inst_ann_ids)
    cat_ids      = list({a['category_id'] for a in inst_anns})
    cats         = coco_inst.loadCats(cat_ids)
    cat_names    = [c['name'] for c in cats]

    print(f"\n--- imgidx={imgidx} | idx73k={idx73k} | cocoId={coco_id} | split={coco_split} ---")
    print(f"Categories: {cat_names}")
    print("Captions:")
    for ann in anns:
        print(f"  - {ann['caption']}")