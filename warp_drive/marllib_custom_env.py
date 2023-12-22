import argparse
import os
from envs.crowd_sim.crowd_sim import RLlibCrowdSim
from warp_drive.utils.common import get_project_root
from train_rl_policy import LARGE_DATASET_NAME
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY

# register all scenario with env class
REGISTRY = {}
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default=LARGE_DATASET_NAME, help="Dataset to use for training."
)
if __name__ == '__main__':
    args = parser.parse_args()
    # register new env
    ENV_REGISTRY["crowdsim"] = RLlibCrowdSim
    # initialize env
    if args.dataset == LARGE_DATASET_NAME:
        from datasets.Sanfrancisco.env_config import BaseEnvConfig
    else:
        from datasets.KAIST.env_config import BaseEnvConfig
    env = marl.make_env(environment_name="crowdsim", map_name=LARGE_DATASET_NAME,
                        abs_path=os.path.join(get_project_root(), "run_configs", "mcs_data_collection.yaml"),
                        env_params=BaseEnvConfig)
    # pick mappo algorithms
    mappo = marl.algos.mappo(hyperparam_source="test")
    # customize model
    model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "512-512"})
    # start learning
    mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 10000000}, local_mode=True, num_gpus=1,
              num_workers=2, share_policy='all', checkpoint_freq=100)
