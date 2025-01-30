"""
    All function for plotting
"""
import os
import re
import string

import matplotlib

within_parentheses = r'\([^()]*\)'

matplotlib.use('pgf')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

output_chinese = False
FONTSIZE = 10
FIG_SIZE = (17, 4)
MARKER_SIZE = 6
MARKER_WIDTH = 2
MARKER_FACE_COLOR = 'none'
LINE_WIDTH = 4
colors = ['red', 'goldenrod', 'forestgreen', 'teal', 'violet', 'grey', 'turquoise']
markers = ['o', '*', '^', 'v', 'd', 's', '+', 'x']

X_TICKS = 'x-ticks'
OURS = 'DRL-MTUCS'
RL_SOTA = 'HAPPO'
GCRL_SOTA = 'AIM'
MCS_SOTA = 'DRL-EMS'
TRADITIONAL = 'mTSP'
if output_chinese:
    RANDOM = '随机'
    I_emer = "紧急任务有效处理比率 ($\mathit{I}_{\mathrm{emer}}$)"
    I_surv = "监控任务有效处理比率 ($\mathit{I}_{\mathrm{surv}}$)"
    eta = "能耗比率 (\scalebox{2}{$\eta$})"
    I_index = "有效任务处理指数 ($\mathit{I}$)"
    X_BLUR = "最大模糊要求 ($\delta\scriptstyle\mathrm{max}$)"
    X_UAV = "群体数量 (U)"
    X_SURV_THRE = "监控任务阈值 ($\mathrm{AoI}_\mathrm{th}^\mathrm{surv}$)"
    X_TASK_TYPE = "任务类型数量"
else:
    RANDOM = 'Random'
    I_emer = "Valid Handling Ratio For Emergency ($\mathit{I}_{\mathrm{emer}}$)"
    I_surv = "Valid Handling Ratio For Surveillance ($\mathit{I}_{\mathrm{surv}}$)"
    eta = "Energy Consumption Ratio ($\eta$)"
    I_index = "Valid Task Handling Index ($\mathit{I}$)"
    X_BLUR = "Maximum Image Blur Requirement ($\delta\scriptstyle\mathrm{max}$)"
    X_UAV = "No. of UAVs ($\mathit{U}$)"
    X_SURV_THRE = "AoI Threshold For Surveillance Tasks ($\mathrm{AoI}_\mathrm{th}^\mathrm{surv}$)"
    X_TASK_TYPE = "No. of Task Types"

DATASET = 'dataset'
DATAS = 'datas'
Y_RANGE = 'yrange'
San = 'SanFrancisco'
Chengdu = 'Chengdu'
EPS = 'eps'
generate_dir = os.path.join('/Users', 'Charlie', 'Desktop', 'MCS_graph', 'zh' if output_chinese else 'en')
all_methods = [OURS, TRADITIONAL, RL_SOTA, MCS_SOTA, GCRL_SOTA, RANDOM]

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker


def compare_plot(x_label, y_label, x, yrange, data_dict, method_order, eps=0.3, ax=None):
    """
    :param str x_label: The x axis label of the plot
    :param str y_label: The y axis label of the plot
    :param list x: The x axis values
    :param list[int] yrange: The range of y axis
    :param dict[str, list[double]] data_dict: dictionary with key as data label, and actual data as values
    :param str dataset:
    :param float eps: the range for y axis
    :param ax: The axis to plot on (for subplots)
    """

    # If ax is passed, use that axis, otherwise create a new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.set_ylabel(re.sub(within_parentheses, '', y_label))
    ax.set_xticks(x)
    margin = (yrange[1] - yrange[0]) * eps
    ax.set_ylim(max(0, (yrange[0] - margin)), yrange[1] + margin)
    if (yrange[1] - yrange[0]) < 0.2:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    else:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    # Plot each algorithm's data
    for index, data_name in enumerate(method_order):
        assert len(data_dict[data_name]) == len(x), (f"Data length should be equal to x={x}, "
                                                     f"get {data_dict[data_name]}")
        ax.plot(x, data_dict[data_name], color=colors[index], marker=markers[index],
                label=data_name, markersize=MARKER_SIZE,
                markeredgewidth=MARKER_WIDTH, markerfacecolor=MARKER_FACE_COLOR)

    ax.grid(True, linestyle='--')

    return ax


def generate_plots(x_label: str, x: list, data_dicts: dict, dataset: str):
    """
    Generate plots for each dataset
    :param str x_label: The x axis label of the plot
    :param list x: The x axis values
    :param dict[str, dict[str, list[double]]] data_dicts: dictionary with key as data label, and actual data as values
    :param str dataset: The dataset name
    """

    # Update Matplotlib parameters
    plt.rcParams.update(
        {
            "xtick.labelsize": FONTSIZE + 6,
            "ytick.labelsize": FONTSIZE + 6,
            'axes.labelsize': FONTSIZE + 8,
            "text.usetex": True,
            'font.family': 'serif',
            'pgf.texsystem': 'xelatex',
            'pgf.preamble': r'''
                \usepackage{xeCJK}
                \usepackage{amsmath, amssymb}
                \renewcommand{\rmdefault}{ptm}
                \renewcommand{\sfdefault}{phv}
                \renewcommand{\ttdefault}{pcr}
            ''',
            "pgf.rcfonts": False,
        }
    )

    if EPS in data_dicts:
        eps = data_dicts.pop(EPS)
    else:
        eps = 0.1

    n_plots = len(data_dicts)

    # Create subplots
    fig, axes = plt.subplots(1, n_plots, figsize=FIG_SIZE, sharex=True)

    # If there's only one subplot, make axes an iterable
    if n_plots == 1:
        axes = [axes]

    for idx, (y_label, data_dict) in enumerate(data_dicts.items()):
        if EPS in data_dict:
            eps = data_dict.pop(EPS)
        all_values = np.concatenate(list(data_dict.values()))
        y_range = [np.nanmin(all_values), np.nanmax(all_values)]
        if os.path.exists(generate_dir) is False:
            os.makedirs(generate_dir)
        dataset_sub_dir = os.path.join(generate_dir, dataset)
        if os.path.exists(dataset_sub_dir) is False:
            os.makedirs(dataset_sub_dir)
        # Generate each plot on its corresponding subplot
        ax = compare_plot(x_label=x_label,
                          y_label=y_label,
                          x=x,
                          yrange=y_range,
                          method_order=all_methods,
                          data_dict=data_dict,
                          eps=eps,
                          ax=axes[idx])

        # Collect handles and labels for the unified legend
    # strip $, \, {, } using re
    latex_exclude = r'[\$\{\}\\0-9]|scalebox|small|scriptstyle|mathit|mathrm'
    raw_x_label = re.sub(latex_exclude, '', x_label)
    # Unified legend from last ax
    handles_, labels_ = ax.get_legend_handles_labels()


    subplot_step = 1 / (n_plots * 2)
    letters = string.ascii_letters
    extra_artists = []
    for index, y_label in zip(range(n_plots), data_dicts.keys()):
        symbol_label = re.findall(within_parentheses, y_label)[0]
        extra_artists.append(fig.text(subplot_step + index * (subplot_step * 2),
                                      -0.03, '('+letters[index]+') ' + symbol_label[1:-1], size=FONTSIZE + 10))
    extra_artists.append(fig.text(0.5, -0.12, x_label, ha='center', size=FONTSIZE + 10))
    extra_artists.append(fig.legend(handles_, labels_, loc='upper center', ncol=len(labels_),
                               fontsize=FONTSIZE + 10, bbox_to_anchor=(0.5, 1.17)))
    # Save the plot
    dataset_sub_dir = os.path.join(generate_dir, dataset)
    if not os.path.exists(dataset_sub_dir):
        os.makedirs(dataset_sub_dir)
    fig.tight_layout()
    fig.savefig(os.path.join(dataset_sub_dir, f"{raw_x_label}.pdf"), backend='pgf',
                bbox_extra_artists=extra_artists,  bbox_inches='tight')
    plt.close()



def calculate_index(all_methods, data_dict):
    new_data_dict = {I_index: {}, **data_dict}
    for method in all_methods:
        # assert method in data_dict[I_emer][DATAS], f"method {method} should be in data_dict"
        sur_ratio = data_dict[I_surv][method]
        eme_ratio = data_dict[I_emer][method]
        energy_ratio = data_dict[eta][method]
        new_data_dict[I_index][method] = (np.minimum(sur_ratio, eme_ratio) / energy_ratio).tolist()
    return new_data_dict


if __name__ == '__main__':

    all_data = {
        # 3.5 / 6, aim, happo are not recorded. MCS_SOTA has one dataset.
        X_UAV: {
            X_TICKS: [2, 3, 4, 5, 7, 10, 20],
            San: {
                I_emer: {
                    OURS: [0.5923, 0.7579, 0.8919, 0.9719, 0.973, 0.9958, 0.9982],
                    # 7,10 not converged for RL_SOTA.
                    RL_SOTA: [0.2684, 0.2902, 0.3505, 0.3768, 0.433, 0.4614, 0.5961],
                    GCRL_SOTA: [0.2551, 0.3158, 0.3449, 0.3561, 0.4558, 0.5182, 0.6817],
                    MCS_SOTA: [0.233, 0.2568, 0.3512, 0.36, 0.4288, 0.5853, 0.7652],
                    TRADITIONAL: [
                        0.484848484,
                        0.545454545,
                        0.606060606,
                        0.666666667,
                        0.696969697,
                        0.818181818,
                        0.929824561,
                    ],
                    RANDOM: [
                        0.210526316,
                        0.228070175,
                        0.245614035,
                        0.333333333,
                        0.298245614,
                        0.280701754,
                        0.49122807,
                    ],
                },
                I_surv: {
                    OURS: [0.746, 0.8302, 0.8836, 0.9224, 0.9474, 0.9639, 0.9785],
                    RL_SOTA: [0.8078, 0.8619, 0.8926, 0.9136, 0.9302, 0.94, 0.9463],
                    GCRL_SOTA: [0.7086, 0.8479, 0.88, 0.8911, 0.9169, 0.9254, 0.9537],
                    MCS_SOTA: [0.6686, 0.79, 0.8426, 0.8698, 0.8915, 0.9293, 0.9632],
                    TRADITIONAL: [
                        0.563333334,
                        0.696666667,
                        0.723333333,
                        0.786666667,
                        0.83,
                        0.833333333,
                        0.98
                    ],
                    RANDOM: [
                        0.486666667,
                        0.503948439,
                        0.546666667,
                        0.503333333,
                        0.506666667,
                        0.593333333,
                        0.886666667
                    ],
                },
                eta: {
                    OURS: [0.6679, 0.6675, 0.6548, 0.6572, 0.6471, 0.5869, 0.5752],
                    RL_SOTA: [0.6717, 0.6641, 0.6503, 0.6499, 0.6466, 0.613, 0.5781],
                    GCRL_SOTA: [0.6750, 0.6753, 0.6598, 0.6166, 0.6166, 0.6041, 0.5857],
                    MCS_SOTA: [0.6641, 0.6675, 0.6693, 0.6688, 0.6704, 0.6663, 0.6591],
                    TRADITIONAL: [
                        0.674018421,
                        0.673658769,
                        0.676202787,
                        0.665288572,
                        0.660551537,
                        0.666067925,
                        0.669078633
                    ],
                    RANDOM: [
                        0.659841383,
                        0.670475665,
                        0.665712714,
                        0.673109421,
                        0.665769289,
                        0.675672576,
                        0.665043771
                    ],
                }
            },
            Chengdu: {
                I_emer: {
                    OURS: [0.6007, 0.7326, 0.9547, 0.9765, 0.986, 0.9979, 0.9992],
                    RL_SOTA: [0.2537, 0.3175, 0.3418, 0.3754, 0.48, 0.5098, 0.6579],
                    GCRL_SOTA: [0.5467, 0.6898, 0.8761, 0.913, 0.3838, 0.5130, 0.7112],
                    MCS_SOTA: [0.2702, 0.3723, 0.3744, 0.4116, 0.4895, 0.5737, 0.7652],
                    TRADITIONAL: [
                        0.363636363,
                        0.424242424,
                        0.566666667,
                        0.636363636,
                        0.666666667,
                        0.909090909,
                        0.964912281,
                    ],
                    RANDOM: [
                        0.280701754,
                        0.263157895,
                        0.298245614,
                        0.368421053,
                        0.315789474,
                        0.368421053,
                        0.578947368,
                    ],

                },
                I_surv: {
                    OURS: [0.7729, 0.8923, 0.9449, 0.9785, 0.99, 0.9947, 0.9988],
                    RL_SOTA: [0.8604, 0.9193, 0.9455, 0.9739, 0.9873, 0.992, 0.9873],
                    GCRL_SOTA: [0.6694, 0.8252, 0.8692, 0.9567, 0.9652, 0.9594, 0.9948],
                    MCS_SOTA: [0.7313, 0.8382, 0.8847, 0.9312, 0.9702, 0.995, 0.9891],
                    TRADITIONAL: [
                        0.646736596,
                        0.71,
                        0.74,
                        0.763333333,
                        0.943333333,
                        0.87,
                        0.986666667,
                    ],
                    RANDOM: [
                        0.36,
                        0.563333333,
                        0.623333333,
                        0.673333333,
                        0.78,
                        0.75,
                        0.983333333
                    ],
                },
                eta: {
                    EPS: 0.5,
                    OURS: [0.6696, 0.666, 0.6643, 0.6571, 0.651, 0.6229, 0.5793],
                    RL_SOTA: [0.6786, 0.6796, 0.6654, 0.6534, 0.6523, 0.6539, 0.5668],
                    GCRL_SOTA: [0.6824, 0.6799, 0.6629, 0.6629, 0.6632, 0.6672, 0.6369],
                    MCS_SOTA: [0.6767, 0.6753, 0.6612, 0.6611, 0.658, 0.6538, 0.6535],
                    TRADITIONAL: [
                        0.671788702,
                        0.661090532,
                        0.658536493,
                        0.653623624,
                        0.657083523,
                        0.655491053,
                        0.657729088
                    ],
                    RANDOM: [
                        0.666898342,
                        0.66149163,
                        0.65589873,
                        0.653389177,
                        0.653389177,
                        0.666898342,
                        0.658762692
                    ],
                }
            }
        },
        X_SURV_THRE: {
            X_TICKS: [15, 20, 25, 30, 35, 40],
            San: {
                I_emer: {
                    OURS: [0.8115, 0.8368, 0.9439, 0.9453, 0.9123, 0.9221],
                    RL_SOTA: [0.3593, 0.3449, 0.3242, 0.2958, 0.3768, 0.3646],
                    GCRL_SOTA: [0.3358, 0.3481, 0.3691, 0.3358, 0.3239, 0.3446],
                    MCS_SOTA: [0.3407, 0.3467, 0.3414, 0.3702, 0.3512, 0.3161],
                    TRADITIONAL: [0.606060606] * 6,
                    RANDOM: [0.245614035] * 6,
                },
                I_surv: {
                    OURS: [0.6717, 0.7309, 0.8161, 0.8401, 0.8602, 0.8939],
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
                    OURS: [0.6652, 0.6618, 0.6575, 0.6547, 0.6517, 0.6487],
                    RL_SOTA: [0.6717, 0.6641, 0.6638, 0.657, 0.6578, 0.65],
                    GCRL_SOTA: [0.6669, 0.6608, 0.6623, 0.6594, 0.6593, 0.6557],
                    MCS_SOTA: [0.6663, 0.6664, 0.6654, 0.666, 0.6693, 0.6664],
                    TRADITIONAL: [0.676202787] * 6,
                    RANDOM: [0.665712714] * 6,

                }
            },
            Chengdu: {
                I_emer: {
                    OURS: [0.9537, 0.9765, 0.9768, 0.9825, 0.9849, 0.9909],
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
                    OURS: [0.6669, 0.6622, 0.6614, 0.6578, 0.6512, 0.6502],
                    RL_SOTA: [0.6791, 0.6746, 0.6753, 0.6777, 0.6788,  0.6734],
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
                    OURS: [0.3861, 0.6558, 0.7933, 0.8364, 0.9491],
                    RL_SOTA: [0.2739, 0.2388, 0.3297, 0.2655, 0.2467],
                    GCRL_SOTA: [0.1733, 0.2618, 0.2739, 0.2582, 0.2521],
                    MCS_SOTA: [0.257, 0.2667, 0.263, 0.2733, 0.3024],
                    TRADITIONAL: [
                        0.333333333,
                        0.333333333,
                        0.363636364,
                        0.393939394,
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
                    OURS: [0.8497, 0.8248, 0.8362, 0.8699, 0.8746],
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
                    OURS: [0.6567, 0.6486, 0.6515, 0.6463, 0.6237],
                    RL_SOTA: [0.6594, 0.664, 0.6614, 0.6637, 0.654],
                    GCRL_SOTA: [0.6706, 0.6684, 0.6621, 0.6541, 0.6502],
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
                    OURS: [0.4285, 0.7309, 0.8212, 0.8842, 0.9176],
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
                    OURS: [0.8959, 0.9169, 0.9192, 0.9304, 0.9461],
                    RL_SOTA: [0.9138, 0.943, 0.9511, 0.9324, 0.9428],
                    GCRL_SOTA: [0.8836, 0.9274, 0.9361, 0.9424, 0.9316],
                    MCS_SOTA: [0.8023, 0.8713, 0.8797, 0.8846, 0.8917],
                    TRADITIONAL: [
                        0.516666667,
                        0.61,
                        0.696666667,
                        0.773333333,
                        0.78,

                    ],
                    RANDOM: [
                        0.266666667,
                        0.28,
                        0.266666667,
                        0.31,
                        0.376666667
                    ],
                },
                eta: {
                    OURS: [0.6618, 0.6636, 0.6599, 0.6543, 0.6521],
                    RL_SOTA: [0.6766, 0.6738, 0.6738, 0.6611, 0.6519],
                    GCRL_SOTA: [0.6732, 0.6724, 0.6724, 0.6616, 0.657],
                    MCS_SOTA: [0.6601, 0.6609, 0.6573, 0.6574,  0.6565],
                    TRADITIONAL: [
                        0.651617832,
                        0.652784691,
                        0.649464292,
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
    for x_label, array_data in all_data.items():
        assert X_TICKS in array_data, "x-ticks should be in array_data"
        xticks = array_data.pop(X_TICKS)
        if EPS in array_data:
            eps = array_data.pop(EPS)
        else:
            eps = None
        for dataset, dataset_data in array_data.items():
            dataset_data = calculate_index(all_methods, dataset_data)
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