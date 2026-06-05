from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class AmtlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    beta_reg: float = 0.25
    max_reg_scale: float = 1.0
    use_blended_actor_update: bool = True
    ppo_blend_weight: float = 0.8
    amtl_blend_weight: float = 0.2
    blend_schedule: str = "constant"
    blend_transition_start: int = 5000
    blend_transition_end: int = 15000
    use_amp: bool = False
    amp_style_weight: float = 0.5
    amp_task_weight: float = 0.5
    amp_discriminator_lr: float = 1.0e-4
    amp_grad_penalty_weight: float = 10.0
    amp_replay_buffer_size: int = 200000
    amp_batch_size: int = 4096
    amp_as_separate_objective: bool = False


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 25
    experiment_name = "beyond_mimic"
    empirical_normalization = True
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = AmtlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        beta_reg=0.25,
        max_reg_scale=1.0,
        use_blended_actor_update=True,
        ppo_blend_weight=0.8,
        amtl_blend_weight=0.2,
        blend_schedule="constant",
        blend_transition_start=5000,
        blend_transition_end=15000,
        use_amp=False,
        amp_style_weight=0.5,
        amp_task_weight=0.5,
        amp_discriminator_lr=1.0e-4,
        amp_grad_penalty_weight=10.0,
        amp_replay_buffer_size=200000,
        amp_batch_size=4096,
        amp_as_separate_objective=False,
    )


LOW_FREQ_SCALE = 0.5


@configclass
class BaseLowFreqPPORunnerCfg(BasePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)
