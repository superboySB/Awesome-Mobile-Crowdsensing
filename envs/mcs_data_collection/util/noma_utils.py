import numpy as np
import math

def w2db(x):
    return 10 * math.log(x, 10)

def db2w(x):
    return math.pow(10, x/10)

def compute_LoS_prob(env_config, dis, height):
    '''
    :param dis: poi和uav之间的3D距离
    :param height: uav的高度
    :return: 建立LoS信道的概率
    '''
    psi = env_config['psi']
    beta = env_config['beta']
    theta = np.arcsin(height / dis) / np.pi * 180  # theta为角度制

    LoS_prob = 1 / (1 + psi * np.exp(-beta * (theta - psi)))
    return LoS_prob


def compute_channel_gain_G2A(env_config, dis, aA=None):
    '''
    :param dis: poi和uav之间或uav和ugv之间的3D距离
    :return: G2A信道的信道增益
    '''
    height = env_config['uav_init_height']
    if aA is None:
        aA = env_config['aA']
    nLoS = db2w(env_config['nLoS'])
    nNLoS = db2w(env_config['nNLoS'])
    Los_prob = compute_LoS_prob(env_config, dis, height)
    channel_gain_G2A = Los_prob * nLoS * dis ** (-aA) + \
                       (1 - Los_prob) * nNLoS * dis ** (-aA)
    return channel_gain_G2A


def compute_channel_gain_G2G(env_config, dis, aG=None):
    '''
    :param dis: poi和ugv之间的2D距离
    :param aG: 可以人为设置aG
    :return: G2G信道的信道增益
    '''
    if aG is None:
        aG = env_config['aG']
    channel_gain_G2G = np.clip(dis, a_max=np.inf, a_min=1.0) ** (-aG)  
    # 4/21 TSP时距离可能为0, 不能加个小常数因为吞吐率会爆炸，合理的是设距离最近为1
    return channel_gain_G2G

def compute_capacity_G2A(env_config, i_dis, j_dis, co_factor=1):
    '''
    :param i_dis: poi i和uav之间的3D距离
    :param j_dis: poi j和uav之间的3D距离
    :param co_factor: 多个无人机同时为一个无人车转发时，平分带宽，隐含地认为无人机的带宽总量固定
    :return: G2A信道的容量
    '''
    n0 = env_config['noise0_density']
    B0 = env_config['bandwidth_subchannel'] / co_factor
    P = env_config['p_poi']
    Gi_G2A = compute_channel_gain_G2A(env_config, i_dis)
    Gj_G2A = compute_channel_gain_G2A(env_config, j_dis) if j_dis != -1 else 0
    sinr = Gi_G2A * P / (n0 * B0 + Gj_G2A * P)
    Ri_G2A = B0 * np.log2(1 + sinr)
    return sinr, Ri_G2A


def compute_capacity_RE(env_config, uav_ugv_dis, i_2d_dis, j_2d_dis, co_factor=1):
    '''
    :param uav_ugv_dis: uav和ugv之间的3D距离
    :param i_2d_dis: poi i和ugv之间的2D距离，deprecated
    :param j_2d_dis: poi j和ugv之间的2D距离
    :return: RE信道的容量
    '''
    n0 = env_config['noise0_density']
    B0 = env_config['bandwidth_subchannel'] / co_factor
    P_poi = env_config['p_poi']  # w
    P_uav = env_config['p_uav']  # w
    Gi_G2G = compute_channel_gain_G2G(env_config, i_2d_dis)
    Gj_G2G = compute_channel_gain_G2G(env_config, j_2d_dis) if j_2d_dis != -1 else 0
    G_RE = compute_channel_gain_G2A(env_config, uav_ugv_dis)  # debug:不可能出现负数
    sinr = (G_RE * P_uav + Gi_G2G * P_poi) / (n0 * B0 + Gj_G2G * P_poi)
    #sinr = (G_RE * P_uav) / (n0 * B0 + Gj_G2G * P_poi)
    Ri_RE = B0 * np.log2(1 + sinr)
    return sinr, Ri_RE


def compute_capacity_G2G(env_config, j_2d_dis, co_factor=1):
    '''
    :param j_2d_dis: poi j 和ugv之间的2D距离
    :return: G2G信道的容量
    '''
    n0 = env_config['noise0_density']
    B0 = env_config['bandwidth_subchannel'] / co_factor
    P = env_config['p_poi']
    Gj_G2G = compute_channel_gain_G2G(env_config, j_2d_dis)
    sinr = Gj_G2G * P / (n0 * B0)
    Rj_G2G = B0 * np.log2(1 + sinr)
    return sinr, Rj_G2G



if __name__=='__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    noma_config = { 
            'noise0_density': 5e-20,
            'bandwidth_subchannel': 20e6/5,
            'p_uav': 3,  # w, 也即34.7dbm
            'p_poi': 0.1,
            'aA': 2,
            'aG': 4,
            'nLoS': 0,  # dB, 也即1w
            'nNLoS': -20,  # dB, 也即0.01w
            'uav_init_height': 100,
            'psi': 9.6,
            'beta': 0.16,
        }
        
    # 定义输入范围
    x = [500*i for i in range(1,10)]# 在0到1000之间生成100个点

    # 计算函数1和函数2的输出
    # y1 = [compute_channel_gain_G2A(noma_config,i) for i in x]
    # y2 = [compute_channel_gain_G2G(noma_config,i) for i in x]

    # print(y1)
    # print(y2)
    
    y1 = [compute_capacity_RE(noma_config, i, -1, -1, co_factor=1)[1]/1e6 for i in x]
    y2 = [compute_capacity_G2G(noma_config, i)[1]/1e6 for i in x]
    # 创建图形并绘制函数1和函数2的曲线
    plt.plot(x, y1, label='G2A')
    print(y1)
    print(y2)
    #plt.plot(x, y2, label='G2G')


    # 添加图例和标签
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')

    # 显示图形
    plt.savefig('dis.png')