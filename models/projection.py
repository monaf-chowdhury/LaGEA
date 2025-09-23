import jax
import jax.numpy as jnp
import functools
import optax
import orbax.checkpoint as ocp
import numpy as np
from flax import linen as nn
from flax.training import train_state
from models import MLP


class Projection(nn.Module):
    def setup(self):
        self.text_encoder = MLP(hidden_dims=(256, 64), activate_final=False)
        self.image_encoder = MLP(hidden_dims=(256, 64), activate_final=False)
        self.feedback_encoder = MLP(hidden_dims=(256, 64), activate_final=False)


    def __call__(self, text_embedding, image_embedding):
        proj_text_embedding = self.text_encoder(text_embedding)
        proj_image_embedding = self.image_encoder(image_embedding)
        return proj_text_embedding, proj_image_embedding

    def encode_image(self, image_embeddings):
        return self.image_encoder(image_embeddings)

    def encode_text(self, text_embedding):
        return self.text_encoder(text_embedding)

    def encode_feedback(self, feedback_embeddings):
        return self.feedback_encoder(feedback_embeddings)

class RewardModel:
    def __init__(self,
                 seed: int = 42,
                 lr: float = 1e-4,
                 margin: float = 0.1,
                 emb_dim: int = 1024,
                 fb_emb_dim: int = 768,         # GPT-2 dims for feedback
                 ckpt_dir: str = None,
                 text_embedding: jnp.ndarray = None,
                 goal_embedding: jnp.ndarray = None,
                 # --- Alignment hyperparams ---
                 tau_bce: float = 0.25,     # temperature for BCE-style diagonal loss
                 tau_nce: float = 0.07,     # temperature for InfoNCE
                 lambda_bce: float = 1.0,   # weight for BCE loss
                 lambda_nce: float = 1.0):  # weight for InfoNCE:
        self.lr = lr
        self.margin = margin
        self.text_embedding = text_embedding
        self.goal_embedding = goal_embedding
        # alignment hyperparameters
        self.tau_bce = tau_bce
        self.tau_nce = tau_nce
        self.lambda_bce = lambda_bce
        self.lambda_nce = lambda_nce

        self.use_goal_delta = True  # When increasing reward shaping (rho) strength

        # --- Hyperparameters for feedback and fused reward ---
        self.gamma_fb   = 0.99     # discount for feedback potential shaping
        self.temp_goal  = 0.05     # temperature/slope for goal potential squashing
        self.temp_fb    = 0.15     # temperature/slope for feedback delta squashing
        self.alpha_base = 0.50     # base weight for feedback vs goal
        self.use_goal_image = True # if you want to blend goal image potential

        # Gating Hyperparameters
        self.alpha_min    = 0.20
        self.alpha_max    = 0.95
        self.agree_floor  = 0.0    # clamp negatives to this before mapping to [0,1]

        self.rng = jax.random.PRNGKey(seed)
        self.rng, key = jax.random.split(self.rng, 2)

        dummy_emb = jnp.ones([1, emb_dim], dtype=jnp.float32)
        dummy_fb      = jnp.ones([1, fb_emb_dim],       dtype=jnp.float32)  # 768 for GPT2, 1024 for LIV

        self.proj = Projection()

        def init_all_encoders(model, text_embedding, image_embeddings, feedback_embeddings):
            _ = model.encode_text(text_embedding)
            _ = model.encode_image(image_embeddings)
            _ = model.encode_feedback(feedback_embeddings)
            return 1.0 # Return a dummy value
        # These keyword arguments now match the function definition above.
        proj_params = self.proj.init(
            {'params': key},
            text_embedding=dummy_emb,
            image_embeddings=dummy_emb,
            feedback_embeddings=dummy_fb,
            method=init_all_encoders
        )['params']
        """
        proj_params = self.proj.init(key,
                                     jnp.ones([1, 1024], dtype=jnp.float32),
                                     dummy_emb)["params"]
        """
        self.proj_state = train_state.TrainState.create(
            apply_fn=self.proj.apply,
            params=proj_params,
            tx=optax.adam(lr))

        if ckpt_dir is not None:
            self.ckpt_dir = ckpt_dir
            self.checkpointer = ocp.StandardCheckpointer()

    @functools.partial(jax.jit, static_argnames=("self",))
    def _l2_normalize(self, x, eps=1e-8):
        return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)

    @functools.partial(jax.jit, static_argnames=("self",))
    def _cos_sim(self, a, b):
        # a, b: (B, D) or (D,) broadcastable; assume not normalized
        a_n = self._l2_normalize(a)
        b_n = self._l2_normalize(b)
        return jnp.sum(a_n * b_n, axis=-1)  # in [-1, 1]
    
    @functools.partial(jax.jit, static_argnames=("self",))
    def _map01(self, x):
        # map any [0,1] scalar/vector to a soft multiplier in [0.5, 1.0]
        return 0.5 + 0.5 * jnp.clip(x, 0.0, 1.0)

    @functools.partial(jax.jit, static_argnames=("self",))
    def _blend_alpha(self, base_alpha, consensus=None, agreement=None, reliability=None):
        """Combine multiple trust signals into a single scalar alpha in [alpha_min, alpha_max]."""
        m = 1.0
        if consensus is not None:
            m = m * self._map01(jnp.asarray(consensus))
        if agreement is not None:
            m = m * self._map01(jnp.asarray(agreement))
        if reliability is not None:
            m = m * self._map01(jnp.asarray(reliability))
        a = base_alpha * m
        # keep a convex blend and within desired operating range
        a = jnp.clip(a, self.alpha_min, self.alpha_max)
        return a
    
    def compute_alpha_used(
        self,
        proj_state,
        feedback_embeddings: jnp.ndarray,
        base_alpha: float,
        consensus: float | None = None,
        reliability: float | None = None,
    ):
        """
        Returns a Python float for the scalar α actually used to mix goal vs feedback
        (with agreement computed on-device). consensus/reliability can be None.
        """
        agree = self.feedback_task_agreement(proj_state, feedback_embeddings)  # jnp scalar in [0,1]
        a = self._blend_alpha(
            base_alpha=jnp.array(base_alpha, dtype=jnp.float32),
            consensus=None if consensus is None else jnp.array(consensus, dtype=jnp.float32),
            agreement=agree,
            reliability=None if reliability is None else jnp.array(reliability, dtype=jnp.float32),
        )
        return float(a)

    @functools.partial(jax.jit, static_argnames=("self",))
    def feedback_task_agreement(self, proj_state, feedback_embeddings):
        """Agreement between instruction text and feedback text → [0,1]."""
        z_txt = self.proj.apply({"params": proj_state.params}, self.text_embedding,     method=self.proj.encode_text)
        z_fb  = self.proj.apply({"params": proj_state.params}, feedback_embeddings,     method=self.proj.encode_feedback)
        cos   = self._cos_sim(z_txt, z_fb)  # can be (B,) or scalar, broadcast ok
        cos   = jnp.mean(cos)
        cos   = jnp.maximum(cos, self.agree_floor)  # avoid penalizing slight negatives too harshly
        return 0.5 * (cos + 1.0)  # → [0,1]

    # ---------- Goal/Instruction potential ----------
    @functools.partial(jax.jit, static_argnames=("self",))
    def get_goal_text_potential(self, proj_state, img_embeddings):
        z_img  = self.proj.apply({"params": proj_state.params}, img_embeddings,      method=self.proj.encode_image)
        z_text = self.proj.apply({"params": proj_state.params}, self.text_embedding, method=self.proj.encode_text)
        cos    = self._cos_sim(z_img, z_text)               # [-1, 1]
        pot01  = 0.5 * (cos + 1.0)                          # [0, 1]
        # optional sharpening for reward scale
        return jnp.tanh((pot01 - 0.5) / self.temp_goal)     # roughly [-1, 1]

    @functools.partial(jax.jit, static_argnames=("self",))
    def get_goal_image_potential(self, proj_state, img_embeddings):
        # If you have a goal image embedding (self.goal_embedding)
        z_img  = self.proj.apply({"params": proj_state.params}, img_embeddings,    method=self.proj.encode_image)
        z_goal = self.proj.apply({"params": proj_state.params}, self.goal_embedding, method=self.proj.encode_image)
        cos    = self._cos_sim(z_img, z_goal)               # [-1, 1]
        pot01  = 0.5 * (cos + 1.0)
        return jnp.tanh((pot01 - 0.5) / self.temp_goal)

    @functools.partial(jax.jit, static_argnames=("self",))
    def get_vlm_reward(self, proj_state, img_embeddings):
        """Backwards-compatible: instruction potential (optionally blended with goal-image potential)."""
        pot_text = self.get_goal_text_potential(proj_state, img_embeddings)
        if hasattr(self, "goal_embedding") and self.goal_embedding is not None and getattr(self, "use_goal_image", False):
            pot_goal = self.get_goal_image_potential(proj_state, img_embeddings)
            # simple average; adjust if you prefer weighted
            return 0.5 * (pot_text + pot_goal)
        return pot_text

    # ---------- Goal delta shaping ----------
    @functools.partial(jax.jit, static_argnames=("self",))
    def get_goal_delta_reward(self, proj_state, img_t, img_tp1):
        # reuse potential (already tanh-bounded)-> take a difference of potentials
        phi_t   = self.get_vlm_reward(proj_state, img_t)
        phi_tp1 = self.get_vlm_reward(proj_state, img_tp1)
        delta = self.gamma_fb * phi_tp1 - phi_t
        return jnp.tanh(delta / (self.temp_goal + 1e-8))

    # ---------- Feedback delta shaping ----------
    @functools.partial(jax.jit, static_argnames=("self",))
    def get_feedback_delta_reward(
        self,
        proj_state,
        img_t,              # (B, D_img)   state t
        img_tp1,            # (B, D_img)   state t+1
        feedback_embeddings,# (B, D_fb) or (1, D_fb)
        key_frame_weights=None,  # (B,) or None
        confidence=None,         # (B,) or scalar in [0,1] or None
        gamma=None               # scalar or None
    ):
        # encode
        z_t   = self.proj.apply({"params": proj_state.params}, img_t,   method=self.proj.encode_image)
        z_tp1 = self.proj.apply({"params": proj_state.params}, img_tp1, method=self.proj.encode_image)
        z_fb  = self.proj.apply({"params": proj_state.params}, feedback_embeddings, method=self.proj.encode_feedback)

        # sims to feedback
        sim_t   = self._cos_sim(z_t,   z_fb)   # (B,)
        sim_tp1 = self._cos_sim(z_tp1, z_fb)   # (B,)

        g = jnp.array(self.gamma_fb if gamma is None else gamma, dtype=jnp.float32)
        delta = g * sim_tp1 - sim_t            # (B,) can be [-2, 2] in theory

        # gates
        w = jnp.ones_like(delta)
        if key_frame_weights is not None:
            w = w * jnp.asarray(key_frame_weights).reshape(-1)
        if confidence is not None:
            # allow scalar or vector confidence in [0,1]
            conf = jnp.asarray(confidence)
            if conf.ndim == 0:
                conf = jnp.ones_like(delta) * conf
            w = w * jnp.clip(conf, 0.0, 1.0)

        # scale to a comfortable range with tanh temperature
        delta_scaled = jnp.tanh(delta / self.temp_fb)  # [-1,1] emphasizing sign & magnitude
        return w * delta_scaled

    # ---------- Confidence-aware fusion ----------
    @functools.partial(jax.jit, static_argnames=("self",))
    def get_fused_reward(
        self,
        proj_state,
        img_t,
        img_tp1,
        feedback_embeddings,
        key_frame_weights=None,
        # confidence is intentionally unused now
        confidence=None,
        alpha=None,
        consensus=None,      # ∈ [0,1], from K-sample self-consistency (host-side)
        reliability=None     # ∈ [0,1], from EMA per error code (host-side)
    ):
        # r_goal = self.get_vlm_reward(proj_state, img_t)  # [-1,1]
        r_goal = ( self.get_goal_delta_reward(proj_state, img_t, img_tp1)
               if getattr(self, "use_goal_delta", False)
               else self.get_vlm_reward(proj_state, img_t) )
    

        # directional feedback delta (already keyframe-gated inside)
        r_fb   = self.get_feedback_delta_reward(
            proj_state, img_t, img_tp1, feedback_embeddings,
            key_frame_weights=key_frame_weights,
            confidence=None   # don't use LLM "confidence"
        )

        # compute agreement on-device to stay JIT-friendly
        agree = self.feedback_task_agreement(proj_state, feedback_embeddings)  # [0,1] scalar

        base = self.alpha_base if (alpha is None) else alpha
        a    = self._blend_alpha(base_alpha=base,
                                consensus=consensus,
                                agreement=agree,
                                reliability=reliability)

        fused = (1.0 - a) * r_goal + a * r_fb
        return r_goal, r_fb, fused, a

    @functools.partial(jax.jit, static_argnames=("self",))
    def train_feedback_step(self,
                            batch_img_embeddings: jnp.ndarray,
                            batch_feedback_embeddings: jnp.ndarray,
                            batch_successes: jnp.ndarray,
                            batch_keyframe_weights: jnp.ndarray,
                            proj_state):
        """
        Hybrid alignment objective with key-frame gating:
          (1) BCE-style diagonal supervision (success -> high, failure -> low),
          (2) InfoNCE on success rows (feedback_i prefers its own image_i vs others),
        both weighted by key-frame weights and modulated by goal similarity.

        Shapes:
          batch_img_embeddings:      (B, D)
          batch_feedback_embeddings: (B, D)
          batch_successes:           (B,) in {0,1}
          batch_keyframe_weights:    (B,) >= 0
        """

        def loss_fn(params):
            # ---- Project & L2-normalize in the shared space ----
            proj_img = self.proj.apply({"params": params}, batch_img_embeddings,      method=self.proj.encode_image)
            proj_fb  = self.proj.apply({"params": params}, batch_feedback_embeddings, method=self.proj.encode_feedback)

            proj_img = proj_img / (jnp.linalg.norm(proj_img, axis=-1, keepdims=True) + 1e-8)
            proj_fb  = proj_fb  / (jnp.linalg.norm(proj_fb,  axis=-1, keepdims=True) + 1e-8)

            # Diagonal cosine (feedback_i vs image_i)
            diag_sim = jnp.sum(proj_img * proj_fb, axis=-1)  # (B,), in [-1,1]

            # Goal similarity to modulate weights (encourages geometry-consistent frames)
            proj_goal = self.proj.apply({"params": params}, self.goal_embedding, method=self.proj.encode_image)
            proj_goal = proj_goal / (jnp.linalg.norm(proj_goal, axis=-1, keepdims=True) + 1e-8)  # (1, D)
            goal_sim  = jnp.sum(proj_img * proj_goal, axis=-1)                # (B,), [-1,1]
            goal_sim01 = (goal_sim + 1.0) * 0.5                               # to [0,1]

            # ---- Build stable sample weights (stop gradients implicitly via JAX) ----
            kf_w = batch_keyframe_weights                                      # (B,)
            base_w = kf_w * (0.5 + 0.5 * goal_sim01)                           # emphasize key, near-goal
            # normalize around ~1 to keep learning rate stable
            mean_w = jnp.mean(base_w) + 1e-8
            samp_w = jnp.clip(base_w / mean_w, 0.0, 3.0)                       # clip long tails

            pos_mask = (batch_successes > 0.5).astype(jnp.float32)             # (B,)
            neg_mask = 1.0 - pos_mask

            # ------------------ (1) BCE-style diagonal supervision ------------------
            # Treat scaled cosine as a logit; push up for success, down for failure
            logits_diag = diag_sim / self.tau_bce                              # (B,)
            targets     = pos_mask                                             # 1 for success, 0 for fail

            bce_vec = optax.sigmoid_binary_cross_entropy(logits_diag, targets) # (B,)
            bce_loss = jnp.sum(bce_vec * samp_w) / (jnp.sum(samp_w) + 1e-8)

            # ------------------ (2) InfoNCE over success rows -----------------------
            # Full similarity matrix (feedback_i x image_j)
            sim_mat   = jnp.einsum("bd,nd->bn", proj_fb, proj_img)             # (B, B), cosine since both normalized
            logits_nce = sim_mat / self.tau_nce
            labels     = jnp.arange(sim_mat.shape[0], dtype=jnp.int32)         # positives on diagonal

            ce_rows = optax.softmax_cross_entropy_with_integer_labels(logits_nce, labels)  # (B,)
            # Only count success rows, and weight by samp_w
            nce_row_w = samp_w * pos_mask
            nce_loss  = jnp.sum(ce_rows * nce_row_w) / (jnp.sum(nce_row_w) + 1e-8)

            # Total
            total = self.lambda_bce * bce_loss + self.lambda_nce * nce_loss

            # ------------- Logging (averages guarded for empty sets) -------------
            avg_pos_sim = jnp.sum(diag_sim * pos_mask) / (jnp.sum(pos_mask) + 1e-8)
            avg_neg_sim = jnp.sum(diag_sim * neg_mask) / (jnp.sum(neg_mask) + 1e-8)

            log_info = {
                "feedback_alignment_loss": total,
                "feedback_bce_loss": bce_loss,
                "feedback_nce_loss": nce_loss,
                "avg_success_similarity": avg_pos_sim,
                "avg_fail_similarity": avg_neg_sim,
                "avg_goal_similarity": jnp.mean(goal_sim),
                "kf_weight_mean": jnp.mean(kf_w),
                "kf_weight_nonzero": jnp.mean((kf_w > 0).astype(jnp.float32)),
            }
            return total, log_info

        valgrad = jax.value_and_grad(loss_fn, has_aux=True)
        (total_loss, log_info), grads = valgrad(proj_state.params)
        new_proj_state = proj_state.apply_gradients(grads=grads)
        return new_proj_state, log_info

  
    def update_feedback_alignment(self, batch):
        """
        Expects batch to have:
          - batch.embeddings            (B, D_img)
          - batch.feedback_embeddings   (B, D_fb)
          - batch.successes             (B,)
          - batch.keyframe_weights      (B,)
        """
        self.proj_state, log_info = self.train_feedback_step(
            batch.embeddings,
            batch.feedback_embeddings,
            batch.successes,
            batch.keyframe_weights,
            self.proj_state
        )
        return log_info
    
    # ===== Weighted symmetric contrastive objectives (drop-in) =====
    @functools.partial(jax.jit, static_argnames=("self", "tau", "label_smoothing", "lam_align", "lam_uniform"),)
    def train_feedback_contrastive_step_weighted(
        self,
        batch,                    # expects .embeddings, .feedback_embeddings, .successes, .keyframe_weights
        proj_state,
        tau: float = 0.07,        # temperature
        label_smoothing: float = 0.0,
        lam_align: float = 0.0,   # alignment reg
        lam_uniform: float = 0.0  # uniformity reg
    ):
        """
        Contrastive Control from feedback:
        - anchors = feedback embeddings
        - positives = per-row matching image embeddings (same row)
        - negatives = all other rows in the mini-batch
        - weighted by key-frame saliency, success, and goal-proximity gates
        - symmetric loss (fb->img and img->fb)
        """

        def _uniformity_penalty(z: jnp.ndarray) -> jnp.ndarray:
            # z: (B, D) L2-normalized features
            # log E[exp(-2 ||zi - zj||^2)]
            # pairwise squared distances via (||a||^2 + ||b||^2 - 2 a·b)
            sim = z @ z.T                              # (B, B)
            # ||zi - zj||^2 = 2 - 2 cos_sim (since ||zi|| = ||zj|| = 1)
            sq_d = 2.0 - 2.0 * jnp.clip(sim, -1.0, 1.0)
            return jnp.log(jnp.mean(jnp.exp(-2.0 * sq_d) + 1e-12) + 1e-12)

        def _one_hot(n, smoothing):
            eye = jnp.eye(n, dtype=jnp.float32)
            if smoothing <= 0.0:
                return eye
            return eye * (1.0 - smoothing) + (smoothing / n)

        def loss_fn(params):
            # --- Project & L2-normalize ---
            z_img = self.proj.apply({"params": params}, batch.embeddings,          method=self.proj.encode_image)
            z_fb  = self.proj.apply({"params": params}, batch.feedback_embeddings, method=self.proj.encode_feedback)

            # (Optional) project goal once for gating
            z_goal = self.proj.apply({"params": params}, self.goal_embedding,      method=self.proj.encode_image)

            # Normalize to unit sphere for cosine
            z_img = z_img / (jnp.linalg.norm(z_img, axis=-1, keepdims=True) + 1e-8)
            z_fb  = z_fb  / (jnp.linalg.norm(z_fb,  axis=-1, keepdims=True) + 1e-8)
            z_goal = z_goal / (jnp.linalg.norm(z_goal, axis=-1, keepdims=True) + 1e-8)

            B = z_img.shape[0]
            # --- Per-example gates/weights ---
            # key-frame weights (already [0,1] and typically normalized per-episode)
            w_kf = jnp.asarray(batch.keyframe_weights).reshape(-1)

            # success gate: 0.5 for fail, 1.0 for success (soft boost to positives)
            succ = jnp.asarray(batch.successes).reshape(-1)
            w_succ = 0.5 + 0.5 * succ

            # goal gate: map cosine to [0,1]
            g_sim = jnp.clip(jnp.sum(z_img * z_goal, axis=-1), -1.0, 1.0)
            g_gate = 0.5 * (g_sim + 1.0)

            # final per-row weight (normalize later)
            w = w_kf * w_succ * g_gate
            w = w / (jnp.sum(w) + 1e-8) * B  # normalize to mean 1 to keep loss scale stable

            # --- Similarity matrices & logits (temperature-scaled) ---
            # fb->img
            sim_f2i = (z_fb @ z_img.T) / jnp.maximum(tau, 1e-8)   # (B, B)
            # img->fb
            sim_i2f = (z_img @ z_fb.T) / jnp.maximum(tau, 1e-8)   # (B, B)

            # numerical stability
            sim_f2i = sim_f2i - jax.lax.stop_gradient(jnp.max(sim_f2i, axis=1, keepdims=True))
            sim_i2f = sim_i2f - jax.lax.stop_gradient(jnp.max(sim_i2f, axis=1, keepdims=True))

            # --- Label smoothed targets ---
            targets = _one_hot(B, label_smoothing)

            # per-row cross-entropy, then weight by w
            ce_f2i = optax.softmax_cross_entropy(sim_f2i, targets)   # (B,)
            ce_i2f = optax.softmax_cross_entropy(sim_i2f, targets)   # (B,,)
            loss_ce = 0.5 * (jnp.sum(w * ce_f2i) + jnp.sum(w * ce_i2f)) / (jnp.sum(w) + 1e-8)

            # --- Tiny regularizers to keep geometry well-behaved ---
            # alignment (pull matched pairs together)
            align = jnp.mean(jnp.sum((z_img - z_fb) ** 2, axis=-1))
            # uniformity on both spaces
            uni = 0.5 * (_uniformity_penalty(z_img) + _uniformity_penalty(z_fb))

            total = loss_ce + lam_align * align + lam_uniform * uni

            # --- Logging ---
            diag_f2i = jnp.diag(sim_f2i)
            # approximate "avg negative" as off-diagonal mean
            neg_mask = 1.0 - jnp.eye(B, dtype=jnp.float32)
            avg_neg = (jnp.sum(sim_f2i * neg_mask) / jnp.maximum(jnp.sum(neg_mask), 1.0))

            log_info = {
                "fb_contrastive_loss": total,
                "fb_contrastive_nce": loss_ce,
                "align_reg": align,
                "uniformity_reg": uni,
                "avg_pos_logit": jnp.mean(diag_f2i),
                "avg_neg_logit": avg_neg,
                "avg_weight": jnp.mean(w),
                "tau": jnp.array(tau, dtype=jnp.float32),
                "gate_success_mean": jnp.mean(w_succ),
                "gate_goal_mean": jnp.mean(g_gate),
                "gate_keyframe_mean": jnp.mean(w_kf),
            }
            return total, log_info

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, log_info), grad = grad_fn(proj_state.params)
        new_proj_state = proj_state.apply_gradients(grads=grad)
        return new_proj_state, log_info

    def update_feedback_contrastive_weighted(
        self,
        batch,
        tau: float = 0.07,
        label_smoothing: float = 0.05,
        lam_align: float = 0.02,
        lam_uniform: float = 1e-3,
    ):
        """
        Public entrypoint. Pass the *same minibatch* you already use elsewhere:
        requires batch.embeddings, batch.feedback_embeddings, batch.successes, batch.keyframe_weights.
        """
        self.proj_state, log_info = self.train_feedback_contrastive_step_weighted(
            batch,
            self.proj_state,
            tau=tau,
            label_smoothing=label_smoothing,
            lam_align=lam_align,
            lam_uniform=lam_uniform,
        )
        return log_info

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_pos_step(self,
                       pos_embeddings,
                       neg_embeddings,
                       lag_embeddings,
                       proj_state):
        def loss_fn(params):
            proj_text_embedding = self.proj.apply(
                {"params": params}, self.text_embedding,
                method=self.proj.encode_text)

            proj_pos_embeddings = self.proj.apply(
                {"params": params}, pos_embeddings,
                method=self.proj.encode_image)
            proj_neg_embeddings = self.proj.apply(
                {"params": params}, neg_embeddings,
                method=self.proj.encode_image)
            proj_lag_embeddings = self.proj.apply(
                {"params": params}, lag_embeddings,
                method=self.proj.encode_image)

            pos_cosine = optax.cosine_similarity(proj_text_embedding,
                                                 proj_pos_embeddings)
            neg_cosine = optax.cosine_similarity(proj_text_embedding, 
                                                 proj_neg_embeddings)
            lag_cosine = optax.cosine_similarity(proj_text_embedding, 
                                                 proj_lag_embeddings)

            # pos-neg: pos_cosine > lag_cosine > negative_cosine
            neg_mask = (neg_cosine - pos_cosine + self.margin) > 0
            neg_loss = neg_mask * (neg_cosine - pos_cosine)

            # pos-pos: pos_cosine > lag_cosine
            pos_mask = (lag_cosine - pos_cosine + self.margin) > 0
            pos_loss = pos_mask * (lag_cosine - pos_cosine)
            total_loss = pos_loss.mean() + neg_loss.mean()
            log_info = {
                "pos_cosine": pos_cosine.mean(),
                "pos_cosine_max": pos_cosine.max(),
                "pos_cosine_min": pos_cosine.min(),

                "neg_cosine": neg_cosine.mean(),
                "neg_cosine_max": neg_cosine.max(),
                "neg_cosine_min": neg_cosine.min(),

                "lag_cosine": lag_cosine.mean(),
                "lag_cosine_max": lag_cosine.max(),
                "lag_cosine_min": lag_cosine.min(),

                "neg_num": neg_mask.sum(),
                "neg_loss": neg_loss.mean(),
                "neg_loss_max": neg_loss.max(),

                "pos_num": pos_mask.sum(),
                "pos_loss": pos_loss.mean(),
                "pos_loss_max": pos_loss.max(),
            }
            return total_loss, log_info
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)        
        (_, log_info), grad = grad_fn(proj_state.params)
        new_proj_state = proj_state.apply_gradients(grads=grad)
        return new_proj_state, log_info

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_neg_step(self,
                       batch,
                       proj_state):
        def loss_fn(params):
            proj_text_embedding = self.proj.apply(
                {"params": params}, self.text_embedding,
                method=self.proj.encode_text)

            proj_embeddings = self.proj.apply(
                {"params": params}, batch.embeddings,
                method=self.proj.encode_image)

            # cosine similarity
            cosine = optax.cosine_similarity(proj_text_embedding, proj_embeddings)
            cosine_delta = cosine.reshape(-1, 1) - cosine.reshape(1, -1)
  
            loss = (nn.relu(-cosine_delta + self.margin) * batch.masks).sum(-1).mean()
            log_info = {"pos_loss": loss, "vlm_rewards": cosine}
            return loss, log_info
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, log_info), grad = grad_fn(proj_state.params)
        new_proj_state = proj_state.apply_gradients(grads=grad)
        return new_proj_state, log_info

    def update_neg(self, batch):
        self.proj_state, log_info = self.train_neg_step(batch, self.proj_state) 
        return log_info

    def update_pos(self, batch):  
        self.proj_state, log_info = self.train_pos_step(batch.pos_embeddings,
                                                        batch.neg_embeddings,
                                                        batch.lag_embeddings,
                                                        self.proj_state) 
        return log_info

    def save(self, cnt):
        self.checkpointer.save(f"{self.ckpt_dir}/{cnt}",
                               {"proj": self.proj_state.params},
                               force=True)

    def load(self, ckpt_dir: str, cnt: int = 0):
        raw_restored = self.checkpointer.restore(f"{ckpt_dir}/{cnt}")
        proj_params = raw_restored["proj"]
        self.proj_state = train_state.TrainState.create(
            apply_fn=self.proj.apply,
            params=proj_params,
            tx=optax.adam(self.lr))
