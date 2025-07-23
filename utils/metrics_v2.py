import os
import numpy as np
import argparse
from tqdm import tqdm
import warnings
from scipy.linalg import sqrtm
import torch.nn.functional as F
import torch
from scipy import stats

warnings.filterwarnings("ignore")

def get_feats(dir):
    feat = []
    for fname in tqdm(sorted(os.listdir(dir)), desc=dir.split("/")[-1]):
        id = fname.split('.')[0]
        data = np.load(os.path.join(dir, fname), allow_pickle=True)
        feat.append(data)
    return np.array(feat)

def calc_moe(data):
    confidence = 0.95
    n = len(data)
    sem = np.std(data, ddof=1) / np.sqrt(len(data))
    margin_of_error = stats.t.ppf((1 + confidence) / 2, n - 1) * sem
    return margin_of_error

def calc_fid(gt, pred):
    assert gt.shape == pred.shape
    B, D = gt.shape
    # gt, pred = gt.reshape(B, -1), pred.reshape(B, -1)

    mu_gt = np.mean(gt, axis=0)
    mu_pred = np.mean(pred, axis=0)
    
    sigma_gt = np.cov(gt, rowvar=False) + 1e-8 * np.eye(D)
    sigma_pred = np.cov(pred, rowvar=False) + 1e-8 * np.eye(D)
    
    diff = mu_gt - mu_pred
    mean_diff = np.dot(diff, diff)
    
    # Compute trace of covariance terms
    covmean = sqrtm(sigma_gt @ sigma_pred)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = mean_diff + np.trace(sigma_gt + sigma_pred - 2 * covmean)
    return fid

def calc_r_precision(cond, pred):
    assert cond.shape == pred.shape
    B, D = cond.shape
    
    r1, r2, r3 = 0.0, 0.0, 0.0

    for i in range(B):
        motion_feats = pred[i]  # Shape (D,)
        
        other_indices = np.delete(np.arange(B), i)  # Indices of other samples
        distractors = cond[other_indices]  # Shape (B-1, D)
        
        selected_indices = np.random.choice(
            distractors.shape[0], 
            size=31, 
            replace=False  # Ensure distinct distractors
        )
        selected_distractors = distractors[selected_indices]  # Shape (31, D)
        
        positive = cond[i][np.newaxis, :]  # Shape (1, D)
        cond_feats = np.concatenate([positive, selected_distractors], axis=0)  # Shape (32, D)
        
        distances = np.linalg.norm(cond_feats - motion_feats, axis=-1)  # Shape (32,)
        
        sorted_indices = np.argsort(distances)
        
        r1 += 1 if 0 in sorted_indices[:1] else 0
        r2 += 1 if 0 in sorted_indices[:2] else 0
        r3 += 1 if 0 in sorted_indices[:3] else 0
    
    r1, r2, r3 = r1/B, r2/B, r3/B
    return r1, r2, r3

def calc_diversity(pred):
    B, D = pred.shape

    Sd = 64
    v0 = pred[np.random.choice(B, size=Sd, replace=False)]
    v1 = pred[np.random.choice(B, size=Sd, replace=False)]

    diversity = np.sum(np.linalg.norm(v0 - v1, axis=-1)).item() / Sd

    return diversity

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='### Calculating Metrics_v2 ###')
    parser.add_argument("--feat_dir", type=str, default="./all_features")
    config = parser.parse_args()

    gt = get_feats(os.path.join(config.feat_dir, "motion"))
    cond = get_feats(os.path.join(config.feat_dir, "audio"))

    n = 32
    output = ""

    # Comparison
    # models = ["mospa", "mospa_latent", "mospa_head", "mospa_diff_100", "mospa_diff_4", "mospa_maskg"] \
    #       + ['edge', 'popdg', 'lodge', 'bailando'] \
    #       + ['gt']
    
    models = ["mospa"]

    for model in models:
        pred = get_feats(os.path.join(config.feat_dir, model))

        fid = calc_fid(gt, pred)
        # print(fid)

        np_r1, np_r2, np_r3, np_diversity = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
        for i in range(n):
            r1, r2, r3 = calc_r_precision(cond, pred)
            diversity = calc_diversity(pred)
            np_r1[i], np_r2[i], np_r3[i], np_diversity[i] = r1, r2, r3, diversity
        
        moe_r1, moe_r2, moe_r3, moe_diversity = \
        calc_moe(np_r1), calc_moe(np_r2), calc_moe(np_r3), calc_moe(np_diversity)
        r1, r2, r3, diversity = np_r1.mean(), np_r2.mean(), np_r3.mean(), np_diversity.mean()

        out_str = f"{model}: R-precision: r1: {r1:.3f} +/- {moe_r1:.3f}; r2: {r2:.3f} +/- {moe_r2:.3f}; r3: {r3:.3f} +/- {moe_r3:.3f} | " + '\n' + \
            f"FID: {fid:.3f} | " + '\n' + \
            f"diversity: {diversity:.3f} +/- {moe_diversity:.3f} | " + '\n'

        print(out_str)
        output += f"{out_str}\n"

        with open(os.path.join(config.feat_dir, f"{model}_metrics_v2.out"), 'w') as f:
            f.write(out_str)
    
    with open("metrics_v2.out", 'w') as f:
        f.write(output)