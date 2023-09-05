from envs.now_in_progress.in_progress import EnvUCS
import numpy as np
import time 

def test(num_uav={'carrier':2,'uav':2}):
    env = EnvUCS({   
        'test_mode': True,
        'debug_mode':False,
        'save_path': '.',
        "seed": 1,
        "num_uav":num_uav,
        'dataset':"KAIST",
    })
    new_obs_n = env.reset()
    total = []
    iteration = 0

    done_count = 0
    poi_collect_list = []

    retdict = {'collection_ratio':[],  'consumption_ratio':[]}
    import time
    start = time.time()
    for i in range(1):

        episode_step = 0
        episode_action = []
        while True:
            actions = {}
            for key in num_uav.keys():
                action = []
                for i in range(num_uav[key]):
                    a = np.random.randint(11)
                    while new_obs_n['mask_{}'.format(key)][i][a] != 1:
                        a = np.random.randint(12)
                    action.append(a)
                actions[key] = action
                if sum(new_obs_n['mask_{}'.format(key)][i]) == 1:
                    print(1)
                  
            #action = [actions[i] for i in range(num_uav)]
            new_obs_n, rew_n, done_n, info_n = env.step(actions)
            episode_action.append(actions)
            obs_n = new_obs_n
            done = done_n
            episode_step += 1
            if done:
                done_count += 1
                end = time.time()
                print('time:',end-start)
                
                poi_collect_list.append(info_n['a_data_collection_ratio'])
                retdict = info_n
                #print(info_n)
                obs_n = env.reset()
                
                end2 = time.time()
                print('time:',end2-end)
                episode_step = 0
                iteration += 1 
                break

    # print('collection_ratio:', np.mean(retdict['collection_ratio']),
    #      'uav consumption_ratio:', np.mean(retdict['uav_consumption_ratio']),
    #      'carrier consumption_ratio:', np.mean(retdict['carrier_consumption_ratio']), )
    print(retdict)
    print('\n')


if __name__ == '__main__':
    import  time
    start = time.time()
    test()
    end = time.time()
    print(start-end)



    
    
