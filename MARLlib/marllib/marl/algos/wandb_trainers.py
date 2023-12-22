from marllib.marl.algos.wandb_trainer_template import build_wandb_trainer
from ray.rllib.agents.a3c.a2c import A2C_DEFAULT_CONFIG
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.agents.a3c.a3c import (validate_config as validate_config_a2c, execution_plan as execution_plan_a2c,
                                      get_policy_class as get_policy_a2c)
from ray.rllib.agents.ppo.ppo import (DEFAULT_CONFIG as PPO_DEFAULT_CONFIG, validate_config as validate_config_ppo,
                                      PPOTFPolicy, get_policy_class as get_policy_ppo,
                                      execution_plan as execution_plan_ppo)
from ray.rllib.agents.dqn.simple_q import (DEFAULT_CONFIG as DEFAULT_CONFIG_SIMPLE_Q, SimpleQTorchPolicy,
                                           execution_plan as execution_plan_simple_q,
                                           get_policy_class as get_policy_class_simple_q)

WandbPPOTrainer = build_wandb_trainer(
    name="PPO",
    default_config=PPO_DEFAULT_CONFIG,
    validate_config=validate_config_ppo,
    default_policy=PPOTFPolicy,
    get_policy_class=get_policy_ppo,
    execution_plan=execution_plan_ppo,
    allow_unknown_configs=True
)

WandbA2CTrainer = build_wandb_trainer(
    name="A2C",
    default_config=A2C_DEFAULT_CONFIG,
    default_policy=A3CTorchPolicy,
    get_policy_class=get_policy_a2c,
    validate_config=validate_config_a2c,
    execution_plan=execution_plan_a2c,
    allow_unknown_configs=True
)

WandbSimpleQTrainer = build_wandb_trainer(
    name="SimpleQTrainer",
    default_policy=SimpleQTorchPolicy,
    get_policy_class=get_policy_class_simple_q,
    execution_plan=execution_plan_simple_q,
    default_config=DEFAULT_CONFIG_SIMPLE_Q,
    allow_unknown_configs=True
)