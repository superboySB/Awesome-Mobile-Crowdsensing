"""
    All function for plotting
"""
import os
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

X_TICKS = 'x-ticks'

FONTSIZE = 54
FIG_SIZE = (15, 15)
MARKER_SIZE = 30
MARKER_WIDTH = 5
MARKER_FACE_COLOR = 'none'
LINE_WIDTH = 4
colors = ['red', 'goldenrod', 'forestgreen', 'teal', 'violet', 'grey', 'turquoise']
markers = ['o', '*', '^', 'v', 'd', 's', '+', 'x']

OURS = 'DRL-MTOCS'
RL_SOTA = 'HAPPO'
GCRL_SOTA = 'OUTPACE'
MCS_SOTA = 'DRL-EMS'
TRADITIONAL = 'mTSP'
RANDOM = 'Random'
San = 'SanFrancisco'
Chengdu = 'Chengdu'
I_emer = "Valid Handling Ratio For Emergency ($I_{\mathrm{emer}}$)"
I_surv = "Valid Handling Ratio For Surveillance ($I_{\mathrm{surv}}$)"
eta = "Energy Consumption Ratio ($\eta$)"
I_index = "Valid Handling Index ($I$)"
DATASET = 'dataset'
DATAS = 'datas'
Y_RANGE = 'yrange'

X_BLUR = "Blur Requirement ($\delta$)"
X_UAV = "No. of UAVs (U)"
X_SURV_THRE = "Surveillance Threshold ($\mathrm{AoI}_\mathrm{th}^\mathrm{surv}$)"
X_TASK_TYPE = "No. of Task Types"
generate_dir = os.path.join('/Users', 'Charlie', 'Desktop', 'MCS_graph')


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
    pdf = PdfPages(os.path.join(dataset_sub_dir, '%s-%s.pdf' % (raw_x_label, raw_y_label)))
    plt.rcParams.update({"text.usetex": True, 'font.size': FONTSIZE})
    plt.figure(figsize=FIG_SIZE)
    plt.xlabel(x_label, fontsize=FONTSIZE)
    plt.ylabel(y_label, fontsize=FONTSIZE)
    for index, data_name in enumerate(data_dict):
        assert len(data_dict[data_name]) == len(x), (f"Data length should be equal to x={x}, "
                                                     f"get {data_dict[data_name]}")
        plt.plot(x, data_dict[data_name], color=colors[index], marker=markers[index],
                 label=data_name, markersize=MARKER_SIZE,
                 markeredgewidth=MARKER_WIDTH, markerfacecolor=MARKER_FACE_COLOR)
    plt.xticks(x, x)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    margin = (yrange[1] - yrange[0]) * eps
    plt.ylim(max(0, yrange[0] - margin), yrange[1] + margin)
    plt.grid(True)
    plt.grid(linestyle='--')
    plt.legend(loc='upper center', fontsize=25, ncol=2, markerscale=0.9)
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    pdf.close()


def generate_plots(x_label: str, x: list, data_dicts: dict, dataset: str):
    for y_label, data_dict in data_dicts.items():
        all_values = np.concatenate(list(data_dict.values()))
        y_range = [np.nanmin(all_values), np.nanmax(all_values)]
        compare_plot(x_label=x_label,
                     y_label=y_label,
                     x=x,
                     yrange=y_range,
                     data_dict=data_dict,
                     dataset=dataset)


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
                    OURS: [0.4453, 0.7579, 0.9088, 0.9667, 0.973, 0.9853],
                    RL_SOTA: [0.2193, 0.2481, 0.2705, 0.274, 0.3382, 0.4144],
                    GCRL_SOTA: [0.2744, 0.2979, 0.3291, 0.3561, 0.9709, 0.494],
                    MCS_SOTA: [0.233, 0.2568, 0.3512, 0.36, 0.4288, 0.5853],
                    TRADITIONAL: [
                        np.NaN,
                        0.333333333,
                        0.333333333,
                        0.363636364,
                        0.333333333,
                        0.393939394,
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
                    OURS: [0.746, 0.8302, 0.8827, 0.8999, 0.9405, 0.9643],
                    RL_SOTA: [0.4112, 0.5379, 0.6257, 0.6969, 0.76, 0.8329],
                    GCRL_SOTA: [0.7636, 0.8244, 0.8728, 0.8911, 0.9062, 0.9383],
                    MCS_SOTA: [0.6686, 0.79, 0.8426, 0.8698, 0.8915, 0.9293],
                    TRADITIONAL: [
                        np.NaN,
                        0.716666667,
                        0.683333333,
                        0.723333333,
                        0.726666667,
                        0.78
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
                    OURS: [0.6728, 0.6675, 0.663, 0.6707, 0.6641, 0.6504],
                    RL_SOTA: [0.6516, 0.6539, 0.6613, 0.6595, 0.6602, 0.6613],
                    GCRL_SOTA: [0.6638, 0.6579, 0.6483, 0.6608, 0.6387, 0.6448],
                    MCS_SOTA: [0.6641, 0.6675, 0.6693, 0.6688, 0.6704, 0.6663],
                    TRADITIONAL: [
                        np.NaN,
                        0.680247268,
                        0.679070123,
                        0.671951513,
                        0.673565624,
                        0.669018085
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
                    OURS: [0.6677, 0.8523, 0.9655, 0.9635, 0.9677, 0.9979],
                    RL_SOTA: [0.2249, 0.2642, 0.2804, 0.3186, 0.373, 0.4646],
                    GCRL_SOTA: [0.2768, 0.3207, 0.8488, 0.966, 0.9811, 0.9989],
                    MCS_SOTA: [0.2702, 0.3723, 0.3744, 0.4116, 0.4895, 0.5737],
                    TRADITIONAL: [
                        np.NaN,
                        0.333333333,
                        0.333333333,
                        0.303030303,
                        0.363636364,
                        0.363636364
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
                    OURS: [0.6902, 0.8671, 0.9673, 0.9617, 0.9847, 0.9941],
                    RL_SOTA: [0.395, 0.5174, 0.5995, 0.6563, 0.7682, 0.8557],
                    GCRL_SOTA: [0.8214, 0.8719, 0.9145, 0.9109, 0.9494, 0.9794],
                    MCS_SOTA: [0.7313, 0.8382, 0.8847, 0.9312, 0.9702, 0.995],
                    TRADITIONAL: [
                        np.NaN,
                        0.726666667,
                        0.583333333,
                        0.513333333,
                        0.673333333,
                        0.803333333
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
                    OURS: [0.6689, 0.6662, 0.6707, 0.6683, 0.6694, 0.6585],
                    RL_SOTA: [0.6432, 0.6445, 0.6472, 0.6485, 0.6469, 0.6453],
                    GCRL_SOTA: [0.6783, 0.6747, 0.6715, 0.666, 0.6148, 0.6672],
                    MCS_SOTA: [0.6538, 0.6612, 0.6611, 0.6753, 0.658, 0.6767],
                    TRADITIONAL: [
                        np.NaN,
                        0.679722934,
                        0.670021298,
                        0.674940914,
                        0.662008449,
                        0.667403004
                    ],
                    RANDOM: [
                        0.628323071,
                        0.625096183,
                        0.65589873,
                        0.66149163,
                        0.653389177,
                        0.666898342
                    ],
                    # MCS_SOTA: []
                }
            }
        },
        # 5 / 6, aim is not recorded.
        X_SURV_THRE: {
            X_TICKS: [15, 20, 25, 30, 35, 40],
            San: {
                I_emer: {
                    OURS: [0.893, 0.9242, 0.8477, 0.9158, 0.9281, 0.926],
                    RL_SOTA: [0.3081, 0.2993, 0.2891, 0.2912, 0.2905, 0.2933],
                    GCRL_SOTA: [0.3358, 0.3481, 0.3691, 0.3358, 0.3239, 0.3446],
                    MCS_SOTA: [0.3407, 0.3467, 0.3414, 0.3702, 0.3512, 0.3161],
                    TRADITIONAL: [0.333333333] * 6,
                    RANDOM: [0.245614035] * 6,
                },
                I_surv: {
                    OURS: [0.487, 0.6599, 0.744, 0.8033, 0.8602, 0.8797],
                    RL_SOTA: [0.3183, 0.4199, 0.4992, 0.5673, 0.6307, 0.6626],
                    GCRL_SOTA: [0.65, 0.7489, 0.8018, 0.8254, 0.8663, 0.8768],
                    MCS_SOTA: [0.5337, 0.675, 0.7655, 0.7919, 0.8426, 0.8479],
                    TRADITIONAL: [
                        0.37,
                        0.483333333,
                        0.533333333,
                        0.566666667,
                        0.683333333,
                        0.713333333
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
                    RL_SOTA: [0.6604, 0.6615, 0.6623, 0.6627, 0.6618, 0.6615],
                    GCRL_SOTA: [0.6608, 0.6593, 0.6594, 0.6557, 0.6623, 0.6669],
                    MCS_SOTA: [0.6663, 0.6664, 0.6654, 0.666, 0.6693, 0.6664],
                    TRADITIONAL: [0.679070123] * 6,
                    RANDOM: [0.665712714] * 6,

                }
            },
            Chengdu: {
                I_emer: {
                    OURS: [0.8414, 0.8474, 0.9572, 0.9589, 0.9646, 0.9396],
                    RL_SOTA: [0.2933, 0.2954, 0.2895, 0.2898, 0.2881, 0.2944],
                    GCRL_SOTA: [0.3951, 0.6747, 0.7677, 0.8765, 0.853, 0.7695],
                    MCS_SOTA: [0.3723, 0.3877, 0.4007, 0.3804, 0.3744, 0.4151],
                    TRADITIONAL: [0.333333333] * 6,
                    RANDOM: [0.298245614] * 6,
                },
                I_surv: {
                    OURS: [0.6086, 0.7649, 0.8557, 0.895, 0.9354, 0.9414],
                    RL_SOTA: [
                        0.2939, 0.3995, 0.475, 0.5377, 0.5984, 0.6344
                    ],
                    GCRL_SOTA: [0.7388, 0.8064, 0.8743, 0.8792, 0.9264, 0.9392],
                    MCS_SOTA: [0.508, 0.6735, 0.8135, 0.836, 0.8847, 0.9121],
                    TRADITIONAL: [
                        0.18,
                        0.32,
                        0.436666667,
                        0.486666667,
                        0.583333333,
                        0.7
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
                    OURS: [0.6699, 0.6694, 0.6677, 0.6656, 0.6679, 0.6659],
                    RL_SOTA: [0.6456, 0.6463, 0.6474, 0.648, 0.6482, 0.6492],
                    GCRL_SOTA: [0.6792, 0.673, 0.6739, 0.6715, 0.6715, 0.6701],
                    MCS_SOTA: [0.662, 0.6652, 0.6624, 0.6614, 0.6652, 0.6654],
                    TRADITIONAL: [0.670021298] * 6,
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
                    RL_SOTA: [
                        0.2061, 0.1764, 0.2079, 0.2121, 0.1958
                    ],
                    GCRL_SOTA: [0.2285, 0.2194, 0.2436, 0.237, 0.2558],
                    MCS_SOTA: [
                        0.257, 0.2667, 0.263, 0.2733, 0.3024
                    ],
                    TRADITIONAL: [
                        0.303030303,
                        0.393939394,
                        0.333333333,
                        0.303030303,
                        0.303030303
                    ],
                    RANDOM: [
                        0.181818182,
                        0.242424242,
                        0.303030303,
                        0.272727273,
                        0.242424242
                    ],

                },
                I_surv: {
                    OURS: [0.8078, 0.7906, 0.805, 0.8224, 0.8539],
                    RL_SOTA: [0.5919, 0.6119, 0.6147, 0.6188, 0.6229],
                    GCRL_SOTA: [0.8218, 0.8541, 0.8445, 0.8651, 0.8635],
                    MCS_SOTA: [0.784, 0.8164, 0.8221, 0.82, 0.8211],
                    TRADITIONAL: [
                        0.563333333,
                        0.65,
                        0.67,
                        0.62,
                        0.62
                    ],
                    RANDOM: [
                        0.623333333,
                        0.623333333,
                        0.646666667,
                        0.623333333,
                        0.626666667
                    ],

                },
                eta: {
                    OURS: [0.6592, 0.6551, 0.6561, 0.6499, 0.6425],
                    RL_SOTA: [0.6602, 0.6624, 0.6619, 0.6613, 0.6604],
                    GCRL_SOTA: [0.6182, 0.636, 0.6196, 0.6395, 0.6366],
                    MCS_SOTA: [0.669, 0.6695, 0.6678, 0.6678, 0.6662],
                    TRADITIONAL: [
                        0.667284578,
                        0.672082128,
                        0.667331067,
                        0.6754929,
                        0.670244691
                    ],
                    RANDOM: [
                        0.674454987,
                        0.674734854,
                        0.673773821,
                        0.673326902,
                        0.671280733
                    ],

                }
            },
            Chengdu: {
                I_emer: {
                    OURS: [0.3842, 0.6479, 0.7794, 0.8091, 0.9818],
                    RL_SOTA: [0.1855, 0.1873, 0.1921, 0.1891, 0.1885],
                    GCRL_SOTA: [0.2982, 0.3248, 0.34, 0.6764, 0.8958],
                    MCS_SOTA: [0.2667, 0.3206, 0.3261, 0.3388, 0.3648],
                    TRADITIONAL: [
                        0.272727273,
                        0.363636364,
                        0.272727273,
                        0.303030303,
                        0.333333333
                    ],
                    RANDOM: [
                        0.181818182,
                        0.181818182,
                        0.181818182,
                        0.212121212,
                        0.181818182
                    ],
                },
                I_surv: {
                    OURS: [0.8451, 0.8516, 0.8788, 0.8763, 0.9179],
                    RL_SOTA: [0.5489, 0.5733, 0.5734, 0.5874, 0.5894],
                    GCRL_SOTA: [0.9044, 0.9262, 0.9386, 0.9372, 0.9374],
                    MCS_SOTA: [0.8023, 0.8713, 0.8797, 0.8846, 0.8917],
                    TRADITIONAL: [
                        0.306666667,
                        0.396666667,
                        0.346666667,
                        0.476666667,
                        0.533333333
                    ],
                    RANDOM: [
                        0.416666667,
                        0.453333333,
                        0.456666667,
                        0.466666667,
                        0.44
                    ],
                },
                eta: {
                    OURS: [0.6669, 0.6593, 0.6581, 0.6402, 0.6538],
                    RL_SOTA: [0.649, 0.6494, 0.6483, 0.6467, 0.64],
                    GCRL_SOTA: [0.6648, 0.6709, 0.6694, 0.6667, 0.6575],
                    MCS_SOTA: [0.6573, 0.6609, 0.6574, 0.6601, 0.6565],
                    TRADITIONAL: [
                        0.646477061,
                        0.651824092,
                        0.653571892,
                        0.65695962,
                        0.658353211
                    ],
                    RANDOM: [
                        0.670718327,
                        0.66689537,
                        0.661131978,
                        0.657251911,
                        0.661198504
                    ],
                }
            }
        }

    }

    all_methods = [OURS, RL_SOTA, GCRL_SOTA, MCS_SOTA, TRADITIONAL, RANDOM]
    for x_label, array_data in all_data.items():
        assert X_TICKS in array_data, "x-ticks should be in array_data"
        xticks = array_data.pop(X_TICKS)
        for dataset, dataset_data in array_data.items():
            calculate_index(all_methods, dataset_data)
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
