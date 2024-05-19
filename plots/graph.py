"""
    All function for plotting
"""
import os
import re
import matplotlib

matplotlib.use('pgf')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

output_chinese = True
FONTSIZE = 40
FIG_SIZE = (15, 15)
MARKER_SIZE = 24
MARKER_WIDTH = 5
MARKER_FACE_COLOR = 'none'
LINE_WIDTH = 4
colors = ['red', 'goldenrod', 'forestgreen', 'teal', 'violet', 'grey', 'turquoise']
markers = ['o', '*', '^', 'v', 'd', 's', '+', 'x']

X_TICKS = 'x-ticks'
OURS = 'DRL-MTOCS'
RL_SOTA = 'HAPPO'
GCRL_SOTA = 'AIM'
MCS_SOTA = 'DRL-EMS'
TRADITIONAL = 'mTSP'
if output_chinese:
    RANDOM = '随机'
    I_emer = "紧急任务有效处理比率 ($\mathit{I}_{\mathrm{emer}}$)"
    I_surv = "监控任务有效处理比率 ($\mathit{I}_{\mathrm{surv}}$)"
    eta = "能耗比率 ($\mathit{\eta}$)"
    I_index = "有效处理指数 ($\mathit{I}$)"
    X_BLUR = "模糊要求 ($\mathit{\delta}$)"
    X_UAV = "无人机数量 (U)"
    X_SURV_THRE = "监控任务阈值 ($\mathrm{AoI}_\mathrm{th}^\mathrm{surv}$)"
    X_TASK_TYPE = "任务类型数量"
else:
    RANDOM = 'Random'
    I_emer = "Valid Handling Ratio For Emergency ($\mathit{I}_{\mathrm{emer}}$)"
    I_surv = "Valid Handling Ratio For Surveillance ($\mathit{I}_{\mathrm{surv}}$)"
    eta = "Energy Consumption Ratio ($\mathit{\eta}$)"
    I_index = "Valid Handling Index ($\mathit{I}$)"
    X_BLUR = "Blur Requirement ($\mathit{\delta}$)"
    X_UAV = "No. of UAVs (U)"
    X_SURV_THRE = "Surveillance Threshold ($\mathrm{AoI}_\mathrm{th}^\mathrm{surv}$)"
    X_TASK_TYPE = "No. of Task Types"

DATASET = 'dataset'
DATAS = 'datas'
Y_RANGE = 'yrange'
San = 'SanFrancisco'
Chengdu = 'Chengdu'

generate_dir = os.path.join('/Users', 'Charlie', 'Desktop', 'MCS_graph', 'zh' if output_chinese else 'en')


def compare_plot(x_label, y_label, x, yrange, data_dict, dataset, eps=0.3):
    """

    :param str x_label: The x axis label of the plot
    :param str y_label: The y axis label of the plot
    :param list x: The x axis values
    :param list[int] yrange: The range of y axis
    :param dict[str, list[double]] data_dict: dictionary with key as data label, and actual data as values
    :param str dataset:
    :param float eps: the range for y axis
    """
    if os.path.exists(generate_dir) is False:
        os.makedirs(generate_dir)
    dataset_sub_dir = os.path.join(generate_dir, dataset)
    if os.path.exists(dataset_sub_dir) is False:
        os.makedirs(dataset_sub_dir)
    # strip $, \, {, } using re
    latex_exclude = r'[\$\{\}\\]'
    raw_x_label = re.sub(latex_exclude, '', x_label)
    raw_y_label = re.sub(latex_exclude, '', y_label)

    plt.rcParams.update(
        {
            "text.usetex": True, 'font.size': FONTSIZE, 'pgf.texsystem': 'xelatex',
            'pgf.preamble': r'\usepackage{xeCJK}\fontsize{36}{42}\selectfont', "pgf.rcfonts": False,
        }
    )
    plt.figure(figsize=FIG_SIZE)
    plt.xlabel(x_label, fontsize=FONTSIZE + 8)
    plt.ylabel(y_label, fontsize=FONTSIZE + 8)
    for index, data_name in enumerate(data_dict):
        assert len(data_dict[data_name]) == len(x), (f"Data length should be equal to x={x}, "
                                                     f"get {data_dict[data_name]}")
        plt.plot(x, data_dict[data_name], color=colors[index], marker=markers[index],
                 label=data_name, markersize=MARKER_SIZE,
                 markeredgewidth=MARKER_WIDTH, markerfacecolor=MARKER_FACE_COLOR)
    plt.xticks(x, x)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    margin = (yrange[1] - yrange[0]) * eps
    plt.ylim(max(0, (yrange[0] - margin)), yrange[1] + margin)
    plt.grid(True)
    plt.grid(linestyle='--')
    plt.legend(loc='upper center', ncol=2, markerscale=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_sub_dir, f"{raw_x_label}-{raw_y_label}.pdf"), backend='pgf')
    plt.close()


EPS = 'eps'


def generate_plots(x_label: str, x: list, data_dicts: dict, dataset: str):
    if EPS in data_dicts:
        eps = data_dicts.pop(EPS)
    else:
        eps = 0.5
    for y_label, data_dict in data_dicts.items():
        if EPS in data_dict:
            eps = data_dict.pop(EPS)
        all_values = np.concatenate(list(data_dict.values()))
        y_range = [np.nanmin(all_values), np.nanmax(all_values)]
        compare_plot(x_label=x_label,
                     y_label=y_label,
                     x=x,
                     yrange=y_range,
                     data_dict=data_dict,
                     dataset=dataset,
                     eps=eps)


def calculate_index(all_methods, data_dict):
    data_dict[I_index] = {}
    for method in all_methods:
        # assert method in data_dict[I_emer][DATAS], f"method {method} should be in data_dict"
        sur_ratio = data_dict[I_surv][method]
        eme_ratio = data_dict[I_emer][method]
        energy_ratio = data_dict[eta][method]
        data_dict[I_index][method] = (np.minimum(sur_ratio, eme_ratio) / energy_ratio).tolist()


if __name__ == '__main__':
    all_data = {
        # 3.5 / 6, aim, happo are not recorded. MCS_SOTA has one dataset.
        X_UAV: {
            X_TICKS: [2, 3, 4, 5, 7, 10],
            San: {
                I_emer: {
                    OURS: [0.5923, 0.7579, 0.8919, 0.9719, 0.973, 0.9958],
                    # 7,10 not converged for RL_SOTA.
                    RL_SOTA: [0.2611, 0.2835, 0.3607, 0.3782, 0.3867, 0.5025],
                    GCRL_SOTA: [0.2551, 0.3158, 0.3449, 0.3561, 0.6158, 0.5182],
                    MCS_SOTA: [0.233, 0.2568, 0.3512, 0.36, 0.4288, 0.5853],
                    TRADITIONAL: [
                        0.575757576,
                        0.545454545,
                        0.606060606,
                        0.666666667,
                        0.696969697,
                        0.818181818,
                    ],
                    RANDOM: [
                        0.210526316,
                        0.228070175,
                        0.245614035,
                        0.333333333,
                        0.298245614,
                        0.280701754,
                    ],
                },
                I_surv: {
                    OURS: [0.746, 0.8302, 0.8836, 0.9224, 0.9474, 0.9639],
                    RL_SOTA: [0.8085, 0.8638, 0.8939, 0.914, 0.76, 0.8329],
                    GCRL_SOTA: [0.7086, 0.8479, 0.88, 0.8911, 0.9169, 0.9254],
                    MCS_SOTA: [0.6686, 0.79, 0.8426, 0.8698, 0.8915, 0.9293],
                    TRADITIONAL: [
                        0.563333334,
                        0.696666667,
                        0.723333333,
                        0.786666667,
                        0.83,
                        0.833333333
                    ],
                    RANDOM: [
                        0.486666667,
                        0.38,
                        0.546666667,
                        0.503333333,
                        0.506666667,
                        0.593333333
                    ],
                },
                eta: {
                    OURS: [0.6679, 0.6675, 0.6548, 0.6572, 0.6471, 0.5869],
                    RL_SOTA: [0.6717, 0.6631, 0.65, 0.648, 0.6602, 0.6613],
                    GCRL_SOTA: [0.6533, 0.658, 0.6629, 0.6608, 0.6403, 0.6041],
                    MCS_SOTA: [0.6641, 0.6675, 0.6693, 0.6688, 0.6704, 0.6663],
                    TRADITIONAL: [
                        0.674018421,
                        0.673658769,
                        0.676202787,
                        0.665288572,
                        0.660551537,
                        0.666067925
                    ],
                    RANDOM: [
                        0.659841383,
                        0.670475665,
                        0.665712714,
                        0.673109421,
                        0.665769289,
                        0.675672576
                    ],
                }
            },
            Chengdu: {
                I_emer: {
                    OURS: [0.6007, 0.7326, 0.9547, 0.9765, 0.986, 0.9979],
                    RL_SOTA: [0.2249, 0.2642, 0.2804, 0.3186, 0.373, 0.4646],
                    GCRL_SOTA: [0.5467, 0.6898, 0.8761, 0.913, 0.9214, 0.4975],
                    MCS_SOTA: [0.2702, 0.3723, 0.3744, 0.4116, 0.4895, 0.5737],
                    TRADITIONAL: [
                        np.NaN,
                        0.424242424,
                        0.666666667,
                        0.636363636,
                        0.666666667,
                        0.909090909
                    ],
                    RANDOM: [
                        0.280701754,
                        0.263157895,
                        0.298245614,
                        0.368421053,
                        0.315789474,
                        0.368421053,
                    ],

                },
                I_surv: {
                    OURS: [0.7729, 0.8923, 0.9449, 0.9785, 0.99, 0.9947],
                    RL_SOTA: [0.8615, 0.9209, 0.9431, 0.9744, 0.7682, 0.8557],
                    GCRL_SOTA: [0.6694, 0.8252, 0.8692, 0.9567, 0.9652, 0.9594],
                    MCS_SOTA: [0.7313, 0.8382, 0.8847, 0.9312, 0.9702, 0.995],
                    TRADITIONAL: [
                        np.NaN,
                        0.763333333,
                        0.71,
                        0.74,
                        0.943333333,
                        0.87,
                    ],
                    RANDOM: [
                        0.36,
                        0.563333333,
                        0.623333333,
                        0.673333333,
                        0.78,
                        0.75,
                    ],
                },
                eta: {
                    EPS: 0.5,
                    OURS: [0.6696, 0.666, 0.651, 0.6643, 0.6571, 0.6229],
                    RL_SOTA: [0.6787, 0.6797, 0.6511, 0.663, 0.6469, 0.6453],
                    GCRL_SOTA: [0.6824, 0.6799, 0.6629, 0.6738, 0.6058, 0.5306],
                    MCS_SOTA: [0.6538, 0.6612, 0.6611, 0.6753, 0.658, 0.6767],
                    TRADITIONAL: [
                        np.NaN,
                        0.653623624,
                        0.657083523,
                        0.655491053,
                        0.671788702,
                        0.661090532
                    ],
                    RANDOM: [
                        0.628323071,
                        0.625096183,
                        0.65589873,
                        0.66149163,
                        0.653389177,
                        0.666898342
                    ],
                }
            }
        },
        X_SURV_THRE: {
            X_TICKS: [15, 20, 25, 30, 35, 40],
            San: {
                I_emer: {
                    OURS: [0.893, 0.9242, 0.8477, 0.9158, 0.9281, 0.926],
                    RL_SOTA: [0.3593, 0.3449, 0.3242, 0.2958, 0.3768, 0.3646],
                    GCRL_SOTA: [0.3358, 0.3481, 0.3691, 0.3358, 0.3239, 0.3446],
                    MCS_SOTA: [0.3407, 0.3467, 0.3414, 0.3702, 0.3512, 0.3161],
                    TRADITIONAL: [0.606060606] * 6,
                    RANDOM: [0.245614035] * 6,
                },
                I_surv: {
                    OURS: [0.487, 0.6599, 0.744, 0.8033, 0.8602, 0.8797],
                    RL_SOTA: [0.6942, 0.7888, 0.8486, 0.8711, 0.896, 0.9062],
                    GCRL_SOTA: [0.65, 0.7489, 0.8018, 0.8254, 0.8663, 0.8768],
                    MCS_SOTA: [0.5337, 0.675, 0.7655, 0.7919, 0.8426, 0.8479],
                    TRADITIONAL: [
                        0.283333333,
                        0.413333333,
                        0.503333333,
                        0.65,
                        0.723333333,
                        0.76
                    ],
                    RANDOM: [
                        0.116666667,
                        0.166666667,
                        0.263333333,
                        0.373333333,
                        0.546666667,
                        0.56
                    ],
                },
                eta: {
                    OURS: [0.6632, 0.6643, 0.6593, 0.6555, 0.6613, 0.6612],
                    RL_SOTA: [0.6717, 0.6641, 0.65, 0.657, 0.6578, 0.6638],
                    GCRL_SOTA: [0.6608, 0.6593, 0.6594, 0.6557, 0.6623, 0.6669],
                    MCS_SOTA: [0.6663, 0.6664, 0.6654, 0.666, 0.6693, 0.6664],
                    TRADITIONAL: [0.676202787] * 6,
                    RANDOM: [0.665712714] * 6,

                }
            },
            Chengdu: {
                I_emer: {
                    OURS: [0.9649, 0.9502, 0.9768, 0.9825, 0.9849, 0.9909],
                    RL_SOTA: [0.3533, 0.3954, 0.3281, 0.4084, 0.4063, 0.3523],
                    GCRL_SOTA: [0.3951, 0.6747, 0.7677, 0.8765, 0.853, 0.7695],
                    MCS_SOTA: [0.3723, 0.3877, 0.4007, 0.3804, 0.3744, 0.4151],
                    TRADITIONAL: [0.666666667] * 6,
                    RANDOM: [0.298245614] * 6,
                },
                I_surv: {
                    OURS: [0.6791, 0.7996, 0.8834, 0.9198, 0.9405, 0.964],
                    RL_SOTA: [
                        0.7355, 0.8486, 0.8924, 0.9336, 0.9572, 0.9653
                    ],
                    GCRL_SOTA: [0.7388, 0.8064, 0.8743, 0.8792, 0.9264, 0.9392],
                    MCS_SOTA: [0.508, 0.6735, 0.8135, 0.836, 0.8847, 0.9121],
                    TRADITIONAL: [
                        0.16,
                        0.28,
                        0.333333333,
                        0.41,
                        0.476666667,
                        0.523333333,
                    ],
                    RANDOM: [
                        0.36,
                        0.423333333,
                        0.446666667,
                        0.476666667,
                        0.623333333,
                        0.643333333
                    ],
                },
                eta: {
                    OURS: [0.6669, 0.6548, 0.6611, 0.6614, 0.6502, 0.6611],
                    RL_SOTA: [0.6604, 0.673, 0.6661, 0.647, 0.6784, 0.6733],
                    GCRL_SOTA: [0.6792, 0.673, 0.6739, 0.6715, 0.6715, 0.6701],
                    MCS_SOTA: [0.662, 0.6652, 0.6624, 0.6614, 0.6652, 0.6654],
                    TRADITIONAL: [0.657083523] * 6,
                    RANDOM: [0.65589873] * 6,
                }
            }
        },
        # 5 / 6, aim is not recorded.
        X_BLUR: {
            X_TICKS: [0.25, 0.5, 0.75, 1, 2],
            San: {
                I_emer: {
                    OURS: [0.3861, 0.6085, 0.7933, 0.8364, 0.9491],
                    RL_SOTA: [0.2739, 0.2388, 0.3297, 0.2655, 0.2467],
                    GCRL_SOTA: [0.1733, 0.2618, 0.2739, 0.2582, 0.2521],
                    MCS_SOTA: [0.257, 0.2667, 0.263, 0.2733, 0.3024],
                    TRADITIONAL: [
                        0.363636364,
                        0.393939394,
                        0.333333333,
                        0.333333333,
                        0.545454545
                    ],
                    RANDOM: [
                        0.212121212,
                        0.212121212,
                        0.242424242,
                        0.242424242,
                        0.242424242
                    ],

                },
                I_surv: {
                    OURS: [0.8078, 0.7906, 0.805, 0.8224, 0.8539],
                    RL_SOTA: [0.8781, 0.8857, 0.8878, 0.8911, 0.8983],
                    GCRL_SOTA: [0.8107, 0.8523, 0.8431, 0.866, 0.8814],
                    MCS_SOTA: [0.784, 0.8164, 0.8221, 0.82, 0.8211],
                    TRADITIONAL: [
                        0.45,
                        0.563333333,
                        0.61,
                        0.673333333,
                        0.673333333,
                    ],
                    RANDOM: [
                        0.516666667,
                        0.536666667,
                        0.543333333,
                        0.543333333,
                        0.556666667
                    ],

                },
                eta: {
                    OURS: [0.6592, 0.6551, 0.6561, 0.6499, 0.6425],
                    RL_SOTA: [0.6594, 0.664, 0.6614, 0.6637, 0.654],
                    GCRL_SOTA: [0.6139, 0.6435, 0.6246, 0.6392, 0.6594],
                    MCS_SOTA: [0.669, 0.6695, 0.6678, 0.6678, 0.6662],
                    TRADITIONAL: [
                        0.666145306,
                        0.659299483,
                        0.661675715,
                        0.665001624,
                        0.659716378
                    ],
                    RANDOM: [
                        0.662615607,
                        0.668396666,
                        0.667386306,
                        0.667210604,
                        0.666560631
                    ],

                }
            },
            Chengdu: {
                I_emer: {
                    OURS: [0.3842, 0.6479, 0.7794, 0.8091, 0.9818],
                    RL_SOTA: [0.2842, 0.2764, 0.3358, 0.283, 0.363],
                    GCRL_SOTA: [0.263, 0.3424, 0.3806, 0.3509, 0.757],
                    MCS_SOTA: [0.2667, 0.3206, 0.3261, 0.3388, 0.3648],
                    TRADITIONAL: [
                        0.393939394,
                        0.454545455,
                        0.545454545,
                        0.545454545,
                        0.606060606
                    ],
                    RANDOM: [
                        0.272727273,
                        0.272727273,
                        0.303030303,
                        0.272727273,
                        0.363636364,
                    ],
                },
                I_surv: {
                    OURS: [0.8451, 0.8516, 0.8788, 0.8763, 0.9179],
                    RL_SOTA: [0.9138, 0.943, 0.9511, 0.9324, 0.9428],
                    GCRL_SOTA: [0.8836, 0.9274, 0.9361, 0.9424, 0.9316],
                    MCS_SOTA: [0.8023, 0.8713, 0.8797, 0.8846, 0.8917],
                    TRADITIONAL: [
                        0.356666667,
                        0.22,
                        0.426666667,
                        0.3,
                        0.34
                    ],
                    RANDOM: [
                        0.406666667,
                        0.483333333,
                        0.38,
                        0.356666667,
                        0.356666667,
                    ],
                },
                eta: {
                    OURS: [0.6669, 0.6593, 0.6581, 0.6402, 0.6538],
                    RL_SOTA: [0.6738, 0.6611, 0.6766, 0.6519, 0.6738],
                    GCRL_SOTA: [0.6616, 0.6732, 0.6724, 0.6724, 0.657],
                    MCS_SOTA: [0.6573, 0.6609, 0.6574, 0.6601, 0.6565],
                    TRADITIONAL: [
                        0.651617832,
                        0.649464292,
                        0.652784691,
                        0.648258959,
                        0.637952169
                    ],
                    RANDOM: [
                        0.673616922,
                        0.673210013,
                        0.666377515,
                        0.667388076,
                        0.666154056
                    ],
                }
            }
        }

    }

    all_methods = [OURS, RL_SOTA, GCRL_SOTA, MCS_SOTA, TRADITIONAL, RANDOM]
    for x_label, array_data in all_data.items():
        assert X_TICKS in array_data, "x-ticks should be in array_data"
        xticks = array_data.pop(X_TICKS)
        if EPS in array_data:
            eps = array_data.pop(EPS)
        else:
            eps = None
        for dataset, dataset_data in array_data.items():
            calculate_index(all_methods, dataset_data)
            if eps is not None:
                dataset_data[EPS] = eps
            generate_plots(
                x_label=x_label,
                x=xticks,
                data_dicts=dataset_data,
                dataset=dataset,
            )
    # output all_data as formatted json
    import json

    with open(os.path.join(generate_dir, 'all_data.json'), 'w') as f:
        json.dump(all_data, f, indent=4)
