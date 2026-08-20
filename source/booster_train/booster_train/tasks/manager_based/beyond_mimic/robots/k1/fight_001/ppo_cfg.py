from isaaclab.utils import configclass
from booster_train.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class PPORunnerCfg(BasePPORunnerCfg):
    max_iterations = 50000
    experiment_name = "k1_fight_001"
    run_name = "actor_amtl_stronger_critic_vcoef2_epochs8"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.amtl_apply_to = "actor"
        self.algorithm.actor_pga_mode = "flat"
        self.algorithm.pga_rank = 16
        self.algorithm.pga_direction_weighting = "factor_strength"
        self.algorithm.pga_match_ppo_grad_norm = True
        self.algorithm.min_action_std = 0.2
        self.algorithm.entropy_coef = 0.005
        self.algorithm.value_loss_coef = 2.0
        self.algorithm.num_learning_epochs = 8
