"""
    All function for plotting
"""
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

X_TICKS = 'x-ticks'

FONTSIZE = 40
FIG_SIZE = (13, 13)
MARKER_SIZE = 30
MARKER_WIDTH = 5
MARKER_FACE_COLOR = 'none'
LINE_WIDTH = 4
colors = ['red', 'goldenrod', 'forestgreen', 'teal', 'violet', 'grey', 'turquoise']
markers = ['o', '*', '^', 'v', 'd', 's', '+', 'x']

OURS = 'DRL-EMACS'
TCTSP = 'TC-TSP'
San = 'SanFrancisco'
Chengdu = 'Chengdu'
I_emer = "Valid Handling Ratio For Emergency (I_emer)"
I_surv = "Valid Handling Ratio For Surveillance (I_surv)"
eta = "Energy Consumption Ratio (eta)"
I_index = "Valid Handling Index"
DATASET = 'dataset'
DATAS = 'datas'
Y_RANGE = 'yrange'

X_BLUR = "Blur Reuqirement (delta)"
X_UAV = "Number of UAVs (U)"
X_SURV_THRE = "Surveillance Threshold (Th)"
X_TASK_TYPE = "Number of Task Types"


def compare_plot(x_label, y_label, x, yrange, data_dict, dataset):
    """

    :param str x_label: The x axis label of the plot
    :param str y_label: The y axis label of the plot
    :param list[int] x: The x axis values
    :param list[int] yrange: The range of y axis
    :param dict[str, list[double]] data_dict: dictionary with key as data label, and actual data as values
    :param str group_name:
    """
    generate_dir = os.path.join('/Users', 'Charlie', 'Desktop', 'MCS_graph')
    if os.path.exists(generate_dir) is False:
        os.makedirs(generate_dir)
    dataset_sub_dir = os.path.join(generate_dir, dataset)
    if os.path.exists(dataset_sub_dir) is False:
        os.makedirs(dataset_sub_dir)
    pdf = PdfPages(os.path.join(dataset_sub_dir, '%s-%s.pdf' % (x_label, y_label)))
    plt.figure(figsize=FIG_SIZE)
    plt.xlabel(x_label, fontsize=FONTSIZE)
    plt.ylabel(y_label, fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    for index, data_name in enumerate(data_dict):
        assert len(data_dict[data_name]) == len(x), "Data length should be equal to x"
        plt.plot(x, data_dict[data_name], color=colors[index], marker=markers[index],
                 label=data_name, markersize=MARKER_SIZE,
                 markeredgewidth=MARKER_WIDTH, markerfacecolor=MARKER_FACE_COLOR)
    plt.xticks(x, x)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.ylim(yrange[0], yrange[1] * 1.2)
    plt.grid(True)
    plt.grid(linestyle='--')
    plt.legend(loc='upper center', fontsize=25, ncol=2, markerscale=0.9)
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    pdf.close()


def generate_plots(x_label: str, x: list, data_dicts: dict, dataset: str):
    for y_label, data_dict in data_dicts.items():
        assert Y_RANGE in data_dict, "yrange should be in data_dict"
        assert DATAS in data_dict, "datas should be in data_dict"
        compare_plot(x_label=x_label,
                     y_label=y_label,
                     x=x,
                     yrange=data_dict[Y_RANGE],
                     data_dict=data_dict[DATAS],
                     dataset=dataset)


def calculate_index(all_methods, data_dict):
    for method in all_methods:
        sur_ratio = data_dict[I_surv][DATAS][method]
        eme_ratio = data_dict[I_emer][DATAS][method]
        energy_ratio = data_dict[eta][DATAS][method]
        data_dict[I_index] = {
            Y_RANGE: [0, 1.5],
            DATAS: {
                OURS: np.minimum(sur_ratio, eme_ratio) / energy_ratio,
            }
        }


if __name__ == '__main__':
    all_data = {}
    all_data[X_BLUR] = {
        X_TICKS: [0.5, 1, 2, 3, 5, 7],
        San: {
            I_emer: {
                Y_RANGE: [0.6, 1],
                DATAS: {
                    OURS: [0.7533, 0.9295, 0.9562, 0.9838, 0.9505, 0.9857],
                }
            },
            I_surv: {
                Y_RANGE: [0.8, 1],
                DATAS: {
                    OURS: [0.8254, 0.8662, 0.8818, 0.8886, 0.896, 0.8894],
                }
            },
            eta: {
                Y_RANGE: [0.5, 0.7],
                DATAS: {
                    OURS: [0.6576, 0.6617, 0.6592, 0.6595, 0.6722, 0.6768],
                }
            }
        },
        Chengdu: {
            I_emer: {
                Y_RANGE: [0.6, 1],
                DATAS: {
                    OURS: [0.7533, 0.9295, 0.9562, 0.9838, 0.9505, 0.9857],
                }
            },
            I_surv: {
                Y_RANGE: [0.8, 1],
                DATAS: {
                    OURS: [0.8254, 0.8662, 0.8818, 0.8886, 0.896, 0.8894],
                }
            },
            eta: {
                Y_RANGE: [0.5, 0.7],
                DATAS: {
                    OURS: [0.6576, 0.6617, 0.6592, 0.6595, 0.6722, 0.6768],
                }
            }
        }
    }
    all_methods = [OURS]

    for x_label, array_data in all_data.items():
        assert X_TICKS in array_data, "x-ticks should be in array_data"
        xticks = array_data.pop(X_TICKS)
        for dataset, dataset_data in array_data.items():
            calculate_index(all_methods, dataset_data)
            generate_plots(x_label=x_label,
                           x=xticks,
                           data_dicts=dataset_data,
                           dataset=dataset,
                           )
