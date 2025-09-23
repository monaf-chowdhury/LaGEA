import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.env_name = "reach-v2-goal-hidden"
    config.camera_id = 2
    config.residual = False
    config.eval_episodes = 100
    config.start_timesteps = 10000
    config.max_timesteps = int(1e6)
    config.decay_timesteps = int(7.5e5)
    config.eval_freq = config.max_timesteps // 100
    config.log_freq = config.max_timesteps // 100
    config.ckpt_freq = config.max_timesteps // 10
    config.lr = 1e-4
    config.seed = 0
    config.tau = 0.01
    config.gamma = 0.99
    config.batch_size = 256
    config.hidden_dims = (256, 256)
    config.initializer = "orthogonal"
    config.exp_name = "lagea"

    # relay
    config.relay_threshold = 2500
    config.expl_noise = 0.2

    # fine-tune
    config.rho = 0.25 
    config.rho_cap = 1.0  
    config.gap = 10
    config.crop = False
    config.l2_margin = 0.25
    config.cosine_margin = 0.25
    config.embed_buffer_size = 20000

    # feedback hyperparameters
    config.alpha_feedback = 0.5
    config.episode_success_threshold = 100

    # Contrastive training
    config.contrastive_start = 25000
    config.contrastive_tau = 0.07
    config.contrastive_label_smoothing = 0.05
    config.contrastive_lambda_align = 0.02
    config.contrastive_lambda_uniform = 1e-3

    # --- Adaptive shaping ---
    config.shaping_target_ratio = 0.20   
    config.shaping_warmup_steps = 20000  
    config.shaping_anneal_end   = 600000 
    config.shaping_scale_cap    = 10.0   
    config.shaping_scale_floor  = 0.1
    config.shaping_clip_per_step = 1.0   
    config.shaping_ema_beta     = 0.1   
    config.shape_only_on_fail   = True   
    config.use_goal_delta       = True   

    config.success_ema_beta = 0.01   
    config.progress_power   = 2.0    
    config.alpha_min        = 0.20   
    config.alpha_max        = 0.95

    return config
