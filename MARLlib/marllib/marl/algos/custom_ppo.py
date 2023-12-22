from marllib.marl.algos.wandb_trainer_template import build_wandb_trainer
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG, validate_config, PPOTFPolicy, get_policy_class, execution_plan

WandbPPOTrainer = build_wandb_trainer(
    name="PPO",
    default_config=DEFAULT_CONFIG,
    validate_config=validate_config,
    default_policy=PPOTFPolicy,
    get_policy_class=get_policy_class,
    execution_plan=execution_plan,
    allow_unknown_configs=True
)
