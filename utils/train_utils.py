import logging
import git
import jax
import numpy as np
from flax.core import FrozenDict
import ml_collections


def get_logger(fname: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=fname,
                        filemode='w',
                        force=True)
    logger = logging.getLogger()
    return logger


def log_git(config):
    config.unlock()
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
    except Exception:
        sha = "unknown"   # no repo available
    config["commit"] = sha


def target_update(params: FrozenDict,
                  target_params: FrozenDict,
                  tau: float) -> FrozenDict:

    def _update(param: FrozenDict, target_param: FrozenDict):
        return tau * param + (1 - tau) * target_param

    updated_params = jax.tree_util.tree_map(_update, params, target_params)
    return updated_params

def get_keyframe(similarities: np.ndarray,
                 max_frames: int = 16,
                 min_dist: int | None = None,
                 w_s: float = 0.5,
                 w_v: float = 0.3,
                 w_a: float = 0.2) -> np.ndarray:
    """
    Deterministic keyframe selection that combines goal similarity, its temporal
    derivatives, and coverage. Returns strictly increasing indices.

    Args:
        similarities: 1D array (T,) — higher means closer to goal.
        max_frames:   target number of keyframes (including endpoints).
        min_dist:     minimal temporal distance between selected frames. If None,
                      it's set to max(1, T // (2 * max_frames)).
        w_s, w_v, w_a: weights for similarity, |velocity|, |acceleration| in the
                       saliency score. They should sum to ~1 but need not exactly.

    Returns:
        np.ndarray of indices (<= max_frames, sorted, unique).
    """
    s = np.asarray(similarities, dtype=np.float32).reshape(-1)
    T = s.shape[0]
    if T == 0:
        return np.array([], dtype=int)
    if T <= max_frames:
        return np.arange(T, dtype=int)

    # --- helpers ---
    def z(x):
        x = np.asarray(x, dtype=np.float32)
        std = x.std()
        return (x - x.mean()) / (std + 1e-8) if std > 1e-8 else np.zeros_like(x)

    # First & last are always in
    selected = {0, T - 1}

    # Derivatives
    v = np.diff(s, prepend=s[0])          # velocity
    a = np.diff(v, prepend=v[0])          # acceleration

    # Saliency components
    s_pos = np.maximum(z(s), 0.0)         # reward frames close to goal
    v_abs = z(np.abs(v))                  # big changes
    a_abs = z(np.abs(a))                  # sharp turns

    score = w_s * s_pos + w_v * v_abs + w_a * a_abs

    # Turning points (where velocity changes sign)
    turn_pts = np.where(np.diff(np.sign(v)) != 0)[0] + 1
    selected.update(turn_pts.tolist())

    # Minimal spacing between keyframes
    if min_dist is None:
        min_dist = max(1, T // (2 * max_frames))

    # Temporal Non-Max Suppression (keep high-score frames far apart)
    order = np.argsort(score)[::-1]
    kept = []

    def far_enough(idx, chosen, d):
        if not chosen:
            return True
        return min(abs(idx - j) for j in chosen) >= d

    # Seed with endpoints first (so they are preserved by NMS)
    base = [0, T - 1]
    for b in base:
        if far_enough(b, kept, min_dist):
            kept.append(b)

    # Add turning points by descending score
    if turn_pts.size > 0:
        tp_order = turn_pts[np.argsort(score[turn_pts])[::-1]]
        for idx in tp_order:
            if far_enough(idx, kept, min_dist):
                kept.append(idx)
                if len(kept) >= max_frames:
                    break

    # Fill by highest-score frames under spacing
    if len(kept) < max_frames:
        for idx in order:
            if far_enough(idx, kept, min_dist):
                kept.append(idx)
                if len(kept) >= int(max_frames * 0.7):  # reserve ~30% for coverage
                    break

    # Coverage fill: farthest-in-time points, weighted by saliency
    if len(kept) < max_frames:
        chosen = np.array(sorted(set(kept)), dtype=int)
        mask = np.ones(T, dtype=bool)
        mask[chosen] = False
        while len(chosen) < max_frames and mask.any():
            # Distance to nearest chosen
            idxs = np.where(mask)[0]
            dists = np.min(np.abs(idxs[:, None] - chosen[None, :]), axis=1)
            # Prefer far + salient
            coverage_score = dists * (1.0 + z(score[idxs]))
            pick = idxs[int(np.argmax(coverage_score))]
            chosen = np.sort(np.append(chosen, pick))
            mask[pick] = False
            # Also enforce spacing locally
            left = max(0, pick - min_dist + 1)
            right = min(T, pick + min_dist)
            mask[left:right] = False
        kept = chosen.tolist()

    kept = sorted(set(kept))
    if len(kept) > max_frames:
        # Uniformly prune extras but keep endpoints
        keep = [kept[0], kept[-1]]
        middle = kept[1:-1]
        stride = max(1, int(np.ceil(len(middle) / (max_frames - 2))))
        keep.extend(middle[::stride][:max_frames - 2])
        kept = sorted(set(keep))

    return np.array(kept, dtype=int)

def keyframe_weights_from_indices(T: int,
                                  indices: np.ndarray,
                                  half_window: int = 2,
                                  triangular: bool = True,
                                  normalize: bool = True) -> np.ndarray:
    """
    Build a length-T weight vector that emphasizes frames near keyframes.
    """
    w = np.zeros(T, dtype=np.float32)
    for k in np.asarray(indices, dtype=int):
        for off in range(-half_window, half_window + 1):
            j = np.clip(k + off, 0, T - 1)
            if triangular:
                w[j] = max(w[j], 1.0 - (abs(off) / (half_window + 1)))
            else:
                w[j] = 1.0
    if normalize and w.sum() > 1e-8:
        w /= w.sum()
    return w

def adaptive_rho_shaping(
    config,
    batch,
    fused,                 # fused feedback signal (B,)
    r_goal, r_fb,          # goal delta (B,), feedback delta (B,)
    t,
    ema_env_abs, ema_vlm_abs, ema_scale,
    success_rate_ema,      # external EMA in [0,1]
    alpha_used=None,       # unused here; kept for signature
):
    # failure-only shaping
    fail_mask = (batch.rewards < 0).astype(np.float32) if getattr(config, "shape_only_on_fail", True) \
                else np.ones_like(batch.rewards, dtype=np.float32)

    fused  = np.asarray(fused,  np.float32)
    r_goal = np.asarray(r_goal, np.float32)
    r_fb   = np.asarray(r_fb,   np.float32)

    # gentle pre-clip to keep deltas tame (uses existing knob)
    if getattr(config, "shaping_clip_per_step", None) is not None:
        c = float(config.shaping_clip_per_step)
        fused = np.clip(fused, -c, +c)

    beta = float(getattr(config, "shaping_ema_beta", 0.2))

    # --- simple progress signal in [0,1] ---
    pos_frac = float(np.mean((r_goal > 0).astype(np.float32)))   # fraction moving toward goal
    progress = max(success_rate_ema, pos_frac**2)                 # small early, rises as behavior improves

    # --- map progress -> rho_eff in [0.05, 0.99) (no new hparams) ---
    rho_min, rho_max = 0.05, 0.99 - 1e-6
    rho_eff = rho_min + (rho_max - rho_min) * (1.0 - progress)

    # convert to scale w.r.t. base rho (so effective weight = rho_static * scale = rho_eff)
    rho_static = float(config.rho)
    scale_target = rho_eff / max(rho_static, 1e-8)
    ema_scale = (1 - beta) * ema_scale + beta * scale_target
    scale = ema_scale

    # EMAs (for sane post-clip + logging)
    denom = np.sum(fail_mask) + 1e-8
    env_abs_now = float(np.sum(np.abs(batch.rewards) * fail_mask) / denom)
    ema_env_abs = (1 - beta) * ema_env_abs + beta * env_abs_now
    vlm_abs_now = float(np.mean(np.abs(r_fb)))
    ema_vlm_abs = (1 - beta) * ema_vlm_abs + beta * vlm_abs_now

    # apply shaping
    shaped = fused * scale * fail_mask

    # post-clip to env scale (no extra knob)
    post_clip = 1.0 * ema_env_abs
    shaped = np.clip(shaped, -post_clip, +post_clip)

    # diagnostics
    dbg = {
        "pre_vlm_abs": float(np.mean(np.abs(fused))),
        "ctrl_vlm_abs": vlm_abs_now,
        "post_vlm_abs": float(np.mean(np.abs(shaped))),
        "env_abs_now": env_abs_now,
        "progress": progress,
    }
    achieved = progress  # reuse this field

    return shaped, rho_eff, achieved, scale, ema_env_abs, ema_vlm_abs, ema_scale, dbg
