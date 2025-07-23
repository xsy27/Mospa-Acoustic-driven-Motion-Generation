import numpy as np
from scipy.spatial import procrustes
from scipy.spatial.transform import Rotation as R
from scipy import stats
from scipy.linalg import sqrtm
from tqdm import tqdm
from data.extract import *

def rigid_transform(gt, pred):
    centroid_gt = np.mean(gt, axis=0)
    centroid_pred = np.mean(pred, axis=0)
    gt_centered = gt - centroid_gt
    pred_centered = pred - centroid_pred

    H = pred_centered.T @ gt_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    t = centroid_gt - R @ centroid_pred
    pred_transformed = (R @ pred.T).T + t

    return pred_transformed

def calc_MPJPE(gt, pred):
    error = np.linalg.norm(pred - gt, axis=-1)
    mpjpe = np.mean(error)

    return mpjpe

def calc_M_MPJPE(gt, pred):
    gt_centered = gt - np.mean(gt, axis=-2, keepdims=True)
    pred_centered = pred - np.mean(pred, axis=-2, keepdims=True)
    m_mpjpe = calc_MPJPE(gt_centered, pred_centered)

    return m_mpjpe

def calc_PA_MPJPE(gt, pred):
    n, nframes, njoints, nfeats = gt.shape

    error = 0
    for i in range(n):
        for j in range(nframes):
            aligned_gt, aligned_pred, disparity = procrustes(gt[i, j], pred[i, j])
            error += calc_MPJPE(gt[i, j], aligned_pred)
    pa_mpjpe = error / n / nframes
    
    return pa_mpjpe

def calc_R_MPJPE(gt, pred):
    n, nframes, njoints, nfeats = gt.shape
    gt_root = gt - gt[:, :, 0:1, :]
    pred_root = pred - pred[:, :, 0:1, :]
    r_mpjpe = calc_MPJPE(gt_root, pred_root)

    # error = 0
    # for i in range(n):
    #     for j in range(nframes):
    #         aligned_pred = rigid_transform(gt[i, j], pred[i, j])
    #         error += calc_MPJPE(gt[i, j], aligned_pred)
    # r_mpjpe = error / n / nframes
    
    return r_mpjpe

def calc_APD(pred):
    n, nframes, njoints, nfeats = pred.shape
    pred = pred.reshape(n, nframes, -1)

    error = 0
    for i in range(n):
        for j in range(n):
            if i == j: continue
            error += np.sqrt(np.sum((pred[j] - pred[i]) ** 2))
    afd = error / n / (n-1)

    return afd

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
    gt, pred = gt.reshape(-1, D), pred.reshape(-1, D)

    mu_gt = np.mean(gt, axis=0)
    mu_pred = np.mean(pred, axis=0)
    
    sigma_gt = np.cov(gt, rowvar=False) + 1e-6 * np.eye(D)
    sigma_pred = np.cov(pred, rowvar=False) + 1e-6 * np.eye(D)
    
    diff = mu_gt - mu_pred
    mean_diff = np.sum(diff ** 2)
    
    # Compute trace of covariance terms
    covmean = sqrtm(np.dot(sigma_gt, sigma_pred))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = mean_diff + np.trace(sigma_gt + sigma_pred - 2 * covmean)
    return fid

def calc_r_precision(cond, pred):
    assert cond.shape == pred.shape
    B, D = cond.shape

    r1, r2, r3 = 0.0, 0.0, 0.0

    for i in range(B):
        motion_feats = pred[i]
        distractors = np.delete(cond, (i), axis=0)[np.random.randint(B-1, size=31)]
        cond_feats = np.concatenate([cond[i][np.newaxis, ...], distractors], axis=0)
        distance = np.linalg.norm(cond_feats - motion_feats, axis=(1, 2))
        ranked_indices = np.argsort(distance)

        r1 += 1 if 0 in ranked_indices[:1] else 0
        r2 += 1 if 0 in ranked_indices[:2] else 0
        r3 += 1 if 0 in ranked_indices[:3] else 0
    
    r1, r2, r3 = r1/B, r2/B, r3/B

    return r1, r2, r3

def calc_diversity(pred):
    B, D = pred.shape

    Sd = 100
    v0 = pred[np.random.randint(B, size=Sd)]
    v1 = pred[np.random.randint(B, size=Sd)]

    diversity = np.sum(np.linalg.norm(v0 - v1, axis=(1, 2))).item() / Sd

    return diversity

# def calc_multimodality(pred):
#     B, D = pred.shape

#     Sm = 10
#     C = B
#     multimodality = 0.0

#     for c in range(C):
#         v0 = pred[np.random.randint(B, size=Sm)]
#         v1 = pred[np.random.randint(B, size=Sm)]
#         mc = np.sum(np.linalg.norm(v0 - v1, axis=(1, 2))).item()
#         multimodality += mc
    
#     multimodality /= (C * Sm * N)

#     return multimodality

def calc_metrics_simple(gt, pred):
    n, nframes, njoints, nfeats = gt.shape
    assert (gt.shape == pred.shape)

    m_mpjpe = calc_M_MPJPE(gt, pred)
    mpjpe = calc_MPJPE(gt, pred)
    pa_mpjpe = calc_PA_MPJPE(gt, pred)
    r_mpjpe = calc_R_MPJPE(gt, pred)
    
    apd = calc_APD(pred)

    result = f"M-MPJPE: {m_mpjpe} | MPJPE: {mpjpe} | PA-MPJPE: {pa_mpjpe} | R-MPJPE: {r_mpjpe} | APD: {apd}"

    return result

def calc_metrics(config, gt, pred):
    n, nframes, njoints, nfeats = gt.shape
    assert (gt.shape == pred.shape)

    m_mpjpe = calc_M_MPJPE(gt, pred)
    mpjpe = calc_MPJPE(gt, pred)
    pa_mpjpe = calc_PA_MPJPE(gt, pred)
    r_mpjpe = calc_R_MPJPE(gt, pred)
    
    apd = calc_APD(pred)

    audio_dim, motion_dim = 2273, 300
    if config.model == 'mospa' or config.model == 'mospa_v2':
        extract_audio_features(config, audio_dim)
        extract_gt_motion_features(config, motion_dim)
    extract_motion_features(config, config.model, motion_dim)

    gt_feat = get_feats(os.path.join(config.feat_dir, "motion"))
    cond_feat = get_feats(os.path.join(config.feat_dir, "audio"))

    n = 32
    model = config.model

    pred_feat = get_feats(os.path.join(config.feat_dir, model)) if model != 'real' else gt_feat

    fid = calc_fid(gt_feat, pred_feat)
    # print(fid)

    np_r1, np_r2, np_r3, np_diversity = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
    for i in range(n):
        r1, r2, r3 = calc_r_precision(cond_feat, pred_feat)
        diversity = calc_diversity(pred_feat)
        # multimodality = calc_multimodality(pred_feat)

        np_r1[i], np_r2[i], np_r3[i], np_diversity[i] = r1, r2, r3, diversity
    
    moe_r1, moe_r2, moe_r3, moe_diversity = \
    calc_moe(np_r1), calc_moe(np_r2), calc_moe(np_r3), calc_moe(np_diversity)
    r1, r2, r3, diversity, multimodality = np_r1.mean(), np_r2.mean(), np_r3.mean(), np_diversity.mean()

    result = f"R-precision: r1: {r1:.3f} +/- {moe_r1:.3f}; r2: {r2:.3f} +/- {moe_r2:.3f}; r3: {r3:.3f} +/- {moe_r3:.3f} | " + '\n' + \
    f"{model}: FID: {fid:.3f} | " + '\n' + \
    f"diversity: {diversity:.3f} +/- {moe_diversity:.3f} | " + '\n' + \
    f"M-MPJPE: {m_mpjpe} | MPJPE: {mpjpe} | PA-MPJPE: {pa_mpjpe} | R-MPJPE: {r_mpjpe} | APD: {apd}"

    return result