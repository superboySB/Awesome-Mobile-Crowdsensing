import argparse
from envs.crowd_sim.crowd_sim import CrowdSim


def main(args):
    env = CrowdSim()

    env.reset()
    ongoing = True

    while ongoing:
        action = {}
        for key,value in env.action_space.items():
            action[key] = value.sample()
        
        # print(action)
        obs, rew, done, info = env.step(action)

        if done['__all__']:
            ongoing = False
            print("Render After")
            env.render(args.output_dir, args.plot_loop, args.moving_line)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--output_dir', type=str, default='./logs.html')
    parser.add_argument('--plot_loop', action='store_true')
    parser.add_argument('--moving_line', action='store_true')

    sys_args = parser.parse_args()
    main(sys_args)
