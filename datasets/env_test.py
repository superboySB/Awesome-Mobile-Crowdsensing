import math


def try_sensing_range(r: float, env_config: object):
    """
    :param r: sensing range
    :param env_config: env_config of target simulation
    """
    config = env_config.env
    p_los = math.exp(
        -config.density_of_human_blockers * config.diameter_of_human_blockers * r * (config.h_b - config.h_rx) / (
                config.h_d - config.h_rx))
    p_nlos = 1 - p_los
    PL_los = config.alpha_los + config.beta_los * 10 * math.log10(
        math.sqrt(r * r + config.h_d * config.h_d)) + config.zeta_los
    PL_nlos = config.alpha_nlos + config.beta_nlos * 10 * math.log10(
        math.sqrt(r * r + config.h_d * config.h_d)) + config.zeta_nlos
    PL = p_los * PL_los + p_nlos * PL_nlos
    CL = PL - config.g_tx - config.g_rx
    print(p_los, p_nlos)
    print(CL)
