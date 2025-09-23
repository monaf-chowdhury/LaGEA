import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import cv2
import time
import clip
import optax
import imageio
import ml_collections
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import csv

import torch
import torchvision.transforms as T

from tqdm import trange
from models import SACAgent, FuRLAgent, RewardModel
from utils import (TASKS, DistanceBuffer, EmbeddingBuffer, log_git, get_keyframe, keyframe_weights_from_indices, adaptive_rho_shaping,
                   get_logger, make_env, load_liv)
from feedback import qwen_feedback
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, GPT2Tokenizer, GPT2Model
###################
# Utils Functions #
###################
def crop_center(config, image):
    x1, x2, y1, y2 = 32, 224, 32, 224
    return image[x1:x2, y1:y2, :]


def eval_policy(agent: SACAgent,
                env: gym.Env,
                eval_episodes: int = 10):
    t1 = time.time()
    eval_reward, eval_success, avg_step = 0, 0, 0
    for i in range(1, eval_episodes + 1):
        obs, _ = env.reset()
        while True:
            avg_step += 1
            action = agent.sample_action(obs, eval_mode=True)
            obs, reward, terminated, truncated, info = env.step(action)
            eval_reward += reward
            if terminated or truncated:
                eval_success += info["success"] 
                break

    eval_reward /= eval_episodes
    eval_success /= eval_episodes

    return eval_reward, eval_success, avg_step, time.time() - t1


def setup_logging(config):
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    # logging
    exp_prefix = f"furl_rho{config.rho}"
    exp_name = f"{exp_prefix}/{config.env_name}/s{config.seed}_{timestamp}"
    os.makedirs(f"logs/{exp_prefix}/{config.env_name}", exist_ok=True)
    exp_info = f"# Running experiment for: {exp_name} #"
    print("#" * len(exp_info) + f"\n{exp_info}\n" + "#" * len(exp_info))
    logger = get_logger(f"logs/{exp_name}.log")

    # add git commit info
    log_git(config)
    logger.info(f"Config:\n{config}\n")

    # set random seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    return exp_name, logger


def setup_exp(config):
    # liv
    transform = T.Compose([T.ToTensor()])
    liv = load_liv()        

    # task description embedding
    with torch.no_grad():
        token = clip.tokenize([TASKS[config.env_name]])         
        text_embedding = liv(input=token, modality="text")      # Getting the text embedding for the task instruction   shape: (1, 1024)
    text_embedding = text_embedding.detach().cpu().numpy()
    data = np.load(f"data/oracle/{config.env_name}/s0_c{config.camera_id}.npz")

    # goal_embedding / text_embedding
    oracle_images = data["images"]      # Shape: (501, 480, 480, 3)
    oracle_success = data["success"]    # Shape(501,)
    oracle_traj_len = np.where(oracle_success)[0][0] + 1  # 84

    # initialize the environment
    env = make_env(config.env_name,
                   seed=config.seed,
                   camera_id=config.camera_id)
    eval_seed = config.seed if "hidden" in config.env_name else config.seed+100
    eval_env = make_env(config.env_name,
                        seed=eval_seed,
                        image_size=480,
                        camera_id=config.camera_id)

    # environment parameter
    obs_dim = env.observation_space.shape[0]        # Value: 39
    act_dim = env.action_space.shape[0]             # value: 4
    max_action = env.action_space.high[0]           # value: 1.0
    goal_image = data["images"][oracle_traj_len-1]  # Shape: (480, 480, 3)
    goal_image = crop_center(config, goal_image)    # Shape: (192, 192, 3)
    processed_goal_image = cv2.cvtColor(goal_image, cv2.COLOR_RGB2BGR)
    processed_goal_image = transform(processed_goal_image)  # Normalized Pytorch tensor -> Shape: torch.Size([3, 192, 192])
    goal_embedding = liv(input=processed_goal_image.to("cuda")[None], modality="vision")
    goal_embedding = goal_embedding.detach().cpu().numpy()  # Shape: (1, 1024)

    # fixed LIV representation projection
    vlm_agent = FuRLAgent(obs_dim=obs_dim,
                          act_dim=act_dim,
                          max_action=max_action,
                          seed=config.seed,
                          tau=config.tau,
                          rho=config.rho,
                          margin=config.cosine_margin,
                          gamma=config.gamma,
                          lr=config.lr,
                          text_embedding=text_embedding,
                          goal_embedding=goal_embedding,
                          hidden_dims=config.hidden_dims)

    # SAC agent
    sac_agent = SACAgent(obs_dim=obs_dim,
                         act_dim=act_dim,
                         max_action=max_action,
                         seed=config.seed,
                         tau=config.tau,
                         gamma=config.gamma,
                         lr=config.lr,
                         hidden_dims=config.hidden_dims)

    # Initialize the reward model
    reward_model = RewardModel(seed=config.seed,
                               text_embedding=text_embedding,
                               goal_embedding=goal_embedding,
                               emb_dim = 1024,
                               fb_emb_dim=768)

    # Replay buffer
    replay_buffer = DistanceBuffer(obs_dim=obs_dim,
                                   act_dim=act_dim,
                                   max_size=int(5e5),
                                   emb_dim = 1024,
                                   fb_dim = 768)
    
    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    # -------- GPT-2 for feedback embeddings --------
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2Model.from_pretrained("gpt2").to("cuda").eval()
    # -----------------------------------------------------

    return (
        transform,
        liv,
        env,
        eval_env,
        vlm_agent,
        sac_agent,
        reward_model,
        replay_buffer,
        goal_image,
        model,
        processor,
        gpt2_tokenizer, gpt2_model,   
    )


#################
# Main Function #
#################
def train_and_evaluate(config: ml_collections.ConfigDict):
    start_time = time.time()

    # logging setup
    exp_name, logger = setup_logging(config)
 
    # experiment setup
    (transform,
     liv,
     env,
     eval_env,
     vlm_agent,
     sac_agent,
     reward_model,
     replay_buffer,
     goal_image,
     model,
     processor,
     gpt2_tokenizer, gpt2_model) = setup_exp(config)

    # reward for untrained agent
    eval_episodes = 1 if "hidden" in config.env_name else 10
    eval_reward, eval_success, _, _ = eval_policy(vlm_agent,
                                                  eval_env,
                                                  eval_episodes)
    logs = [{
        "step": 0,
        "eval_reward": eval_reward,
        "eval_success": eval_success,
    }]
    first_success_step = 0

    # trajectory embedding
    embedding_buffer = EmbeddingBuffer(emb_dim=1024,
                                       gap=config.gap,
                                       max_size=config.embed_buffer_size)
    # Feedback embedding buffer
    mem_path = f"logs/{exp_name}_feedback_memory.jsonl"
    traj_embeddings = np.zeros((500, 1024))
    traj_success = np.zeros(500)

    # relay freqs
    relay_freqs = [50, 100, 150, 200]
    relay_freq = np.random.choice(relay_freqs)
    logger.info(f"Relay freqs: {relay_freqs}\n")

    # start training
    obs, _ = env.reset()
    episodic_buffer = []
    episode_frames = []
    goal = obs[-3:]
    reward, ep_task_reward, ep_vlm_reward = 0, 0, 0
    success_cnt, ep_num, ep_step = 0, 0, 0
    lst_ep_step, lst_ep_task_reward, lst_ep_vlm_reward = 0, 0, 0
    sac_step, vlm_step = 0, 0
    lst_sac_step, lst_vlm_step = 0, 0
    policies = ["vlm", "sac"]
    use_relay = True
    episode_success_count = 0
    pos_cosine = neg_cosine = lag_cosine = 0
    pos_cosine_max = neg_cosine_max = lag_cosine_max = 0
    pos_cosine_min = neg_cosine_min = lag_cosine_min = 0
    neg_num = neg_loss = neg_loss_max = 0
    pos_num = pos_loss = pos_loss_max = 0
    ema_env_abs = 0.0
    ema_vlm_abs = 0.0
    ema_scale   = 1.0
    success_rate_ema = 0.0
    succ_beta = getattr(config, "success_ema_beta", 0.01)  # slow drift

    for t in trange(1, config.max_timesteps + 1):
        if t <= config.start_timesteps:
            action = env.action_space.sample()      # shape(4,)
        else:
            if use_relay:
                if policies[(ep_step//relay_freq)%2] == "vlm":
                    vlm_step += 1
                    action = vlm_agent.sample_action(obs)
                else:
                    sac_step += 1
                    action = sac_agent.sample_action(obs)
                    action_noise = np.random.normal(
                        0, sac_agent.max_action*config.expl_noise, size=sac_agent.act_dim)
                    action = (action + action_noise).clip(
                        -sac_agent.max_action, sac_agent.max_action)
            else:
                vlm_step += 1
                action = vlm_agent.sample_action(obs)
        next_obs, task_reward, terminated, truncated, info = env.step(action) 

        # vision language model reward
        image = env.mujoco_renderer.render(
            render_mode="rgb_array",
            camera_id=config.camera_id).copy()  # Shape: (256, 256, 3)
        image = image[::-1] 
        episode_frames.append(image)            # Store raw frame for Qwen feedback later

        image = crop_center(config, image)
        processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        processed_image = transform(processed_image)
        with torch.no_grad():
            image_embedding = liv(input=processed_image.to("cuda")[None], modality="vision")
        image_embedding = image_embedding.detach().cpu().numpy()    # Shape: (1, 1024)
        l2_distance = np.square(image_embedding - vlm_agent.goal_embedding).sum(-1)**0.5
        vlm_reward = reward_model.get_vlm_reward(reward_model.proj_state, image_embedding).item()

        reward = int(info["success"])
        success_cnt += reward

        traj_embeddings[ep_step] = image_embedding
        traj_success[ep_step] = reward
        ep_step += 1

        if first_success_step == 0 and reward:
            first_success_step = ep_step

        reward = 2*int(info["success"]) - 1     # Success -> 1; Failure -> -1
        episodic_buffer.append((
                            obs, action, next_obs,
                            reward, terminated,
                            image_embedding.squeeze(),  # shape (emb_dim,)
                            float(info["success"]),
                            l2_distance.item()
                            ))

        obs = next_obs
        ep_vlm_reward += vlm_reward
        ep_task_reward += task_reward

        # start a new trajectory
        if terminated or truncated:
            episode_was_successful = (first_success_step > 0)
            if first_success_step > 0:  
                episode_success_count += 1

            frames = np.stack(episode_frames, axis = 0 )    # shape (500, H, W, 3)
            sim_scores = (traj_embeddings[:ep_step] @ vlm_agent.goal_embedding.T).squeeze()
            keyframe_indices = get_keyframe(sim_scores, max_frames=16)
            fb_weights = keyframe_weights_from_indices(T=ep_step, indices=keyframe_indices, half_window=2,
                            triangular=True, normalize=False).astype(np.float32)  
            fb_weights = 0.25 + 0.75 * fb_weights  # floor 0.25, peak ~1.0 
            fb_weights = fb_weights / (fb_weights.mean() + 1e-6)  

            idx = np.linspace(0, frames.shape[0] - 1, num=64, dtype=int) 
            frames = frames[idx]

            feedback_text = qwen_feedback(model, processor, frames, config, 
                                        success=episode_was_successful, model_path="Qwen/Qwen2.5-VL-3B-Instruct",
                                        memory_path=mem_path, exp_name=exp_name)
            
            outcome = feedback_text.get('outcome', 'failure')
            primary_error = feedback_text.get('primary_error') or {} 
            error_code = primary_error.get('code', 'unknown')
            suggested_fix = feedback_text.get('suggested_fix', 'no fix')

            embedding_text = feedback_text.get("summary") or feedback_text.get("suggested_fix") or TASKS[config.env_name]
            if not embedding_text or not isinstance(embedding_text, str):
                embedding_text = f"outcome={outcome}; error={error_code}; fix={suggested_fix}"

            # Feedback embedding
            with torch.no_grad():
                enc = gpt2_tokenizer(embedding_text, return_tensors="pt", 
                                     truncation=True, max_length=128, add_special_tokens=True)
                enc = {k: v.to("cuda") for k, v in enc.items()}  
                out = gpt2_model(**enc)
                hidden = out.last_hidden_state

                # last token (causal “summary”)
                last_tok = hidden[:, -1, :]  

                mask = enc.get("attention_mask", torch.ones(hidden.shape[:2], device=hidden.device))
                maskf = mask.unsqueeze(-1).float()                         
                mean_tok = (hidden * maskf).sum(dim=1) / (maskf.sum(dim=1) + 1e-6)  

                gpt2_emb = 0.5 * last_tok + 0.5 * mean_tok

            feedback_embedding = gpt2_emb.detach().cpu().numpy()  # (1, 768) 
            replay_buffer.add_episode(episodic_buffer, feedback_embedding, fb_weights)

            # reset for next episode
            obs, _ = env.reset()
            episodic_buffer = []    
            episode_frames = []
            goal = obs[-3:]
            lst_ep_step = ep_step
            lst_ep_task_reward = ep_task_reward
            lst_ep_vlm_reward = ep_vlm_reward
            lst_sac_step = sac_step
            lst_vlm_step = vlm_step
            ep_vlm_reward = 0
            ep_task_reward = 0
            sac_step = 0
            vlm_step = 0
            policies = policies[::-1]
            relay_freq = np.random.choice(relay_freqs)
            success_rate_ema = (1 - succ_beta) * success_rate_ema + succ_beta * float(episode_was_successful)

            # save embedding
            if first_success_step == 0:
                for j in range(ep_step):
                    embedding_buffer.add(embedding=traj_embeddings[j],
                                         success=False)
            else:
                for j in range(first_success_step):
                    embedding_buffer.add(embedding=traj_embeddings[j],
                                         success=True,
                                         valid=j>=config.gap)

                for j in range(first_success_step, ep_step):
                    if traj_success[j]:
                        embedding_buffer.add(embedding=traj_embeddings[j],
                                             success=True,
                                             valid=j>=config.gap)
                    else:
                        break

            ep_step = 0
            ep_num += 1
            first_success_step = 0

            if use_relay and embedding_buffer.pos_size >= config.relay_threshold:
                use_relay = False

        # training
        if  t > config.start_timesteps:
            if (success_cnt > 0) and (embedding_buffer.valid_size > 0):
                batch = replay_buffer.sample(config.batch_size)
                embedding_batch = embedding_buffer.sample(config.batch_size)
                    
                # --------------------------- Feedback-driven rep learnings
                feedback_log_info = reward_model.update_feedback_alignment(batch)
                if t >= config.contrastive_start:
                    contrastive_log_info = reward_model.update_feedback_contrastive_weighted(
                        batch, tau = config.contrastive_tau, label_smoothing = config.contrastive_label_smoothing,
                        lam_align = config.contrastive_lambda_align, lam_uniform = config.contrastive_lambda_uniform)
                    # merge logs
                    feedback_log_info.update(contrastive_log_info)

                # --- Build delta-shaped, evidence-gated fused reward (B,)
                r_goal, r_fb, fused, a_jax = reward_model.get_fused_reward(
                    reward_model.proj_state,
                    batch.embeddings.astype(np.float32),
                    batch.next_embeddings.astype(np.float32),
                    batch.feedback_embeddings.astype(np.float32),
                    key_frame_weights=batch.keyframe_weights.astype(np.float32),
                    alpha=config.alpha_feedback,
                    consensus=None, reliability=None,
                )
                alpha_used = reward_model.compute_alpha_used(reward_model.proj_state, batch.feedback_embeddings.astype(np.float32),
                                    base_alpha=config.alpha_feedback, consensus=None, reliability=None)
                
                batch_vlm_rewards, rho_eff, shape_ratio, scale, ema_env_abs, ema_vlm_abs, ema_scale, dbg = adaptive_rho_shaping(
                    config, batch, fused, r_goal, r_fb, t,
                    ema_env_abs, ema_vlm_abs, ema_scale,
                    success_rate_ema=success_rate_ema,
                    alpha_used=alpha_used,
                )

                # Projection (pos/neg/lag) + policy update
                proj_log_info = reward_model.update_pos(embedding_batch)
                log_info = vlm_agent.update(batch, batch_vlm_rewards)

                # Add feedback logs (if any)
                log_info.update(feedback_log_info)

                pos_cosine = proj_log_info["pos_cosine"]
                neg_cosine = proj_log_info["neg_cosine"]
                lag_cosine = proj_log_info["lag_cosine"]
                pos_cosine_max = proj_log_info["pos_cosine_max"]
                neg_cosine_max = proj_log_info["neg_cosine_max"]
                lag_cosine_max = proj_log_info["lag_cosine_max"]
                pos_cosine_min = proj_log_info["pos_cosine_min"]
                neg_cosine_min = proj_log_info["neg_cosine_min"]
                lag_cosine_min = proj_log_info["lag_cosine_min"]
                neg_num = proj_log_info["neg_num"]
                neg_loss = proj_log_info["neg_loss"]
                neg_loss_max = proj_log_info["neg_loss_max"]
                pos_num = proj_log_info["pos_num"]
                pos_loss = proj_log_info["pos_loss"]
                pos_loss_max = proj_log_info["pos_loss_max"]

            # collected zero successful trajectory
            else:
                batch = replay_buffer.sample_with_mask(config.batch_size, config.l2_margin)
                feedback_log_info = reward_model.update_feedback_alignment(batch)
                # Projection on negatives
                proj_log_info = reward_model.update_neg(batch)
                # --- Build components (goal, feedback delta, fused, alpha) ---
                r_goal, r_fb, fused, a_jax = reward_model.get_fused_reward(
                    reward_model.proj_state,
                    batch.embeddings.astype(np.float32),
                    batch.next_embeddings.astype(np.float32),
                    batch.feedback_embeddings.astype(np.float32),
                    key_frame_weights=batch.keyframe_weights.astype(np.float32),
                    alpha=config.alpha_feedback,
                    consensus=None, reliability=None,
                )
                alpha_used = reward_model.compute_alpha_used(reward_model.proj_state, 
                                            batch.feedback_embeddings.astype(np.float32),base_alpha=config.alpha_feedback)
                
                batch_vlm_rewards, rho_eff, shape_ratio, scale, ema_env_abs, ema_vlm_abs, ema_scale, dbg = adaptive_rho_shaping(
                    config, batch, fused, r_goal, r_fb, t,
                    ema_env_abs, ema_vlm_abs, ema_scale,
                    success_rate_ema=success_rate_ema,
                    alpha_used=alpha_used,
                )          

                log_info = vlm_agent.update(batch, batch_vlm_rewards)
                pos_loss = proj_log_info["pos_loss"]

                log_info.update(feedback_log_info)

            # update SAC agent
            if use_relay: _ = sac_agent.update(batch)

        # eval
        if t % config.eval_freq == 0:
            eval_reward, eval_success, _, _ = eval_policy(vlm_agent,
                                                          eval_env,
                                                          eval_episodes)

        # logging
        if t % config.log_freq == 0:
            if t > config.start_timesteps:
                log_info.update({
                    "step": t,
                    "success": reward,
                    "task_reward": lst_ep_task_reward,
                    "vlm_reward": lst_ep_vlm_reward,
                    "eval_reward": eval_reward,
                    "eval_success": eval_success,
                    "batch_reward": batch.rewards.mean(),
                    "batch_reward_max": batch.rewards.max(),
                    "batch_reward_min": batch.rewards.min(),
                    "batch_vlm_reward": batch_vlm_rewards.mean(),
                    "batch_vlm_reward_max": batch_vlm_rewards.max(),
                    "batch_vlm_reward_min": batch_vlm_rewards.min(),
                    "alpha_used": alpha_used,
                    "vlm_scale": scale,
                    "rho_eff": rho_eff,
                    "shape_ratio": shape_ratio,
                    "EMA_env_abs": ema_env_abs,
                    "EMA_vlm_abs" : ema_vlm_abs,
                    "EMA_scale": ema_scale,
                    "pre_vlm_abs": dbg["pre_vlm_abs"],
                    "ctrl_vlm_abs": dbg["ctrl_vlm_abs"],
                    "post_vlm_abs": dbg["post_vlm_abs"],
                    "progress": dbg["progress"],
                    "kf_mean": float(np.mean(batch.keyframe_weights)),
                    "kf_max": float(np.max(batch.keyframe_weights)),                    
                    "time": (time.time() - start_time) / 60
                })
                logger.info(
                    f"\n[T {t//1000}K][{log_info['time']:.2f} min] "
                    f"task_reward: {lst_ep_task_reward:.2f}, "
                    f"vlm_reward: {lst_ep_vlm_reward:.2f}\n"
                    f"\tvlm_reward: {eval_reward:.2f}, vlm_success: {eval_success:.0f}, "
                    f"vlm_step: {lst_vlm_step}\n"
                    f"\tq_loss: {log_info['critic_loss']:.3f}, "
                    f"a_loss: {log_info['alpha_loss']:.3f}, "
                    f"q: {log_info['q']:.3f}, q_max: {log_info['q_max']:.3f}\n"
                    f"\tR: {log_info['batch_reward']:.3f}, "
                    f"Rmax: {log_info['batch_reward_max']:.3f}, "
                    f"Rmin: {log_info['batch_reward_min']:.3f}\n"
                    f"\tvlm_R: {log_info['batch_vlm_reward']:.3f}, "
                    f"vlm_Rmax: {log_info['batch_vlm_reward_max']:.3f}, "
                    f"\talpha_used: {log_info['alpha_used']:.3f}\n"
                    f"vlm_Rmin: {log_info['batch_vlm_reward_min']:.3f}\n"
                    f"\tep_num: {ep_num}, success_cnt: {success_cnt}, "
                    f"success: {reward}\n"
                    f"\tpos_ptr: {embedding_buffer.pos_ptr}, "
                    f"valid_ptr: {embedding_buffer.valid_ptr}, "
                    f"neg_ptr: {embedding_buffer.neg_ptr}\n"
                    f"\tpNum: {pos_num}, pLoss: {pos_loss:.3f}, pLossMax: {pos_loss_max:.3f}\n"
                    f"\tnNum: {neg_num}, nLoss: {neg_loss:.3f}, nLossMax: {neg_loss_max:.3f}\n"
                    f"\tpCos: {pos_cosine:.2f}, pCos_max: {pos_cosine_max:.2f}, "
                    f"pCos_min: {pos_cosine_min:.2f}\n"
                    f"\tnCos: {neg_cosine:.2f}, nCos_max: {neg_cosine_max:.2f}, "
                    f"nCos_min: {neg_cosine_min:.2f}\n"
                    f"\tlCos: {lag_cosine:.2f}, lCos_max: {lag_cosine_max:.2f}, "
                    f"lCos_min: {lag_cosine_min:.2f}\n"
                    f"\tgoal: ({goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}), "
                    f"rvlm_R: {log_info['rvlm_reward']:.4f}\n"
                )
                logs.append(log_info)
            else:
                logs.append({
                    "step": t,
                    "task_reward": lst_ep_task_reward,
                    "vlm_reward": lst_ep_vlm_reward,
                    "eval_reward": eval_reward,
                    "eval_success": eval_success,
                    "time": (time.time() - start_time) / 60,
                })
                logger.info(
                    f"\n[T {t//1000}K][{logs[-1]['time']:.2f} min] "
                    f"task_reward: {lst_ep_task_reward:.2f}, "
                    f"vlm_reward: {lst_ep_vlm_reward:.2f}\n"
                )

        # if eval_success == 1:
        #     print(f">>> Experiment Done. Success. Early stopped")
        #     break

    # save logs
    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"logs/{exp_name}.csv")

    # close env
    env.close()
    eval_env.close()

