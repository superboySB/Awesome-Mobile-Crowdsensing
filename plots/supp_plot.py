import matplotlib
import os
import string
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('pgf')
raw_x_label = 'Intrinsic Coefficient ($\omega$)'
raw_y_label = 'Valid Handling Ratio ($I_{m}$)'
generate_dir = os.path.join('/Users', 'Charlie', 'Desktop', 'MCS_graph')
FONTSIZE = 20
# Data for Emergency and Surveillance
alpha = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
datasets =['San Francisco', 'Chengdu']
n_plots = len(datasets)
buffer_enhance = [0.9195, 0.9737]
improved_emergency = [
 [
    0.8818713450292396,  # alpha 0.0
    0.8871345029239763,  # alpha 0.1
    0.9023391812865494,  # alpha 0.2
    0.9204678362573104,  # alpha 0.3
    0.8970760233918127,  # alpha 0.4
    0.8912280701754386,  # alpha 0.5
    0.8760233918128649,  # alpha 0.6
    0.9146198830409359,  # alpha 0.7
    0.8783625730994148,  # alpha 0.8
    0.840350877192982,   # alpha 0.9
    0.49356725146198793 # alpha 1.0
],
[
    0.9678362573099414,  # alpha = 0.0
    0.9596491228070172,  # alpha = 0.1
    0.967836257309941,   # alpha = 0.2
    0.970175438596491,   # alpha = 0.3
    0.9795321637426898,  # alpha = 0.4
    0.9654970760233914,  # alpha = 0.5
    0.9543859649122802,  # alpha = 0.6
    0.9701754385964908,  # alpha = 0.7
    0.9643274853801164,  # alpha = 0.8
    0.9374269005847954,  # alpha = 0.9
    0.9520467836257306   # alpha = 1.0
]
]
improved_surveillance = [[
    0.8651111111111116,  # alpha 0.0
    0.877555555555556,   # alpha 0.1
    0.8816666666666666,  # alpha 0.2
    0.8942222222222218,  # alpha 0.3
    0.8968888888888884,  # alpha 0.4
    0.8865555555555559,  # alpha 0.5
    0.8802222222222224,  # alpha 0.6
    0.8869999999999997,  # alpha 0.7
    0.8840000000000002,  # alpha 0.8
    0.8629999999999995,  # alpha 0.9
    0.936888888888889    # alpha 1.0
],
[
    0.9563333333333331,  # alpha = 0.0
    0.9437777777777777,  # alpha = 0.1
    0.965111111111111,   # alpha = 0.2
    0.9576666666666666,  # alpha = 0.3
    0.9679999999999999,  # alpha = 0.4
    0.959444444444444,   # alpha = 0.5
    0.9643333333333332,  # alpha = 0.6
    0.9674444444444443,  # alpha = 0.7
    0.9726666666666667,  # alpha = 0.8
    0.9774444444444446,  # alpha = 0.9
    0.9762222222222217   # alpha = 1.0
]
    ]
emergency_values = [
 [
    0.71812865497076,  # alpha 0.0
    0.6690058479532163,  # alpha 0.1
    0.6976608187134506,  # alpha 0.2
    0.7532163742690055,  # alpha 0.3
    0.6912280701754382,  # alpha 0.4
    0.6409356725146196,  # alpha 0.5
    0.7040935672514617,  # alpha 0.6
    0.6233918128654968,  # alpha 0.7
    0.505263157894737,   # alpha 0.8
    0.3421052631578945,   # alpha 0.9
    0.3374269005847952    # alpha 1.0
],
[0.7836, 0.8246, 0.8263, 0.8368, 0.8234, 0.8509, 0.8263, 0.8193, 0.7848, 0.5743, 0.4982]
]
surveillance_values = [
[
    0.7720000000000005,  # alpha 0.0
    0.8301111111111107,  # alpha 0.1
    0.8167777777777773,  # alpha 0.2
    0.8361111111111108,  # alpha 0.3
    0.8684444444444441,  # alpha 0.4
    0.8506666666666668,  # alpha 0.5
    0.9225555555555555,  # alpha 0.6
    0.8991111111111109,  # alpha 0.7
    0.9386666666666671,  # alpha 0.8
    0.9428888888888897,  # alpha 0.9
    0.933                 # alpha 1.0
],
[0.8629, 0.8704, 0.8983, 0.9233, 0.9040, 0.9293, 0.9429, 0.9599, 0.9421, 0.9806, 0.9872]
]
plt.rcParams.update(
    {
        "text.usetex": True,
        'xtick.labelsize': FONTSIZE - 6,
        'ytick.labelsize': FONTSIZE - 6,
        'font.family': 'serif',
        'axes.grid': True,
        'lines.markersize': 6,
        'grid.linestyle': '--',
        'pgf.texsystem': 'xelatex',
        'pgf.preamble': r'''
                \usepackage{xeCJK}
                \usepackage{amsmath, amssymb}
                \renewcommand{\rmdefault}{ptm}
                \renewcommand{\sfdefault}{phv}
                \renewcommand{\ttdefault}{pcr}
            ''',
        "pgf.rcfonts": False,
        'legend.loc': 'lower left',
    }
)



def plot_postprocess(ax, fig, plot_title, raw_y_label):
    subplot_step = 1 / (n_plots * 2)
    letters = string.ascii_letters
    handles_, labels_ = ax.get_legend_handles_labels()
    # ax.set_xlabel(raw_x_label, fontsize=FONTSIZE - 4)
    ax.set_xticks(alpha)  # Ensure all alpha values are on the x-axis
    fig.tight_layout()
    # fig.text(0.5, 1.01, plot_title, ha='center', fontsize=16)
    extra_artists = [fig.legend(handles_, labels_, fontsize=FONTSIZE - 4, bbox_to_anchor=(0.5, 0.97), loc='lower center', ncol=2),
                     fig.text(-0.03, 0.5, raw_y_label, va='center', rotation='vertical', size=FONTSIZE + 4),
                     fig.text(0.5, -0.07, '$\omega$', ha='center', size=FONTSIZE + 4)
                     ]
    for index in range(n_plots):
        extra_artists.append(
            fig.text(subplot_step + index * (subplot_step * 2),
                     -0.03, '(' + letters[index] + ') ', size=FONTSIZE)
        )
    fig.savefig(os.path.join(generate_dir, f"{plot_title}.pdf"), backend='pgf',
                bbox_extra_artists=extra_artists, bbox_inches='tight')


# Plot 1, Trade-off between Emergency and Surveillance Tasks
fig, axes = plt.subplots(1,  len(datasets), sharex=True, figsize=(6 * len(datasets), 3))
for ax, dataset, emergency_list, surveillance_list, hline \
        in zip(axes, datasets, emergency_values, surveillance_values, buffer_enhance):
    # Plot Emergency and Surveillance
    ax.plot(alpha, emergency_list, label='Emergency', color='#824D1B', marker='v', linestyle='--')
    ax.plot(alpha, surveillance_list, label='Surveillance', color='#2B8CBE', marker='o', linestyle='--')
    # ax.axhline(hline, linestyle='dashed', linewidth=1, color='red')
    # ax.text(1, hline, 'Performance Boost', transform=ax.transAxes)
    ax.set_title(dataset, fontsize=FONTSIZE - 4)
    ax.set_ylim(np.min([emergency_list, surveillance_list])*0.9, np.max([emergency_list, surveillance_list]) * 1.1)

plot_postprocess(ax, fig, 'Trade-off between emergency and surveillance tasks', '$I_{m}$')
# strangely, bbox_inches must be added. Otherwise, the legend will be cut off.
# Clear the figure
fig.clear()
fig, axes = plt.subplots(1, len(datasets),  sharex=True, figsize=(6 * len(datasets), 3))
for (ax, dataset, improved_emergency_list, improved_surveillance_list,
     emergency_list) in zip(axes, datasets, improved_emergency, improved_surveillance, emergency_values):
    # Plot Improved Emergency and Original Emergency
    # ax.plot(alpha, improved_surveillance_list, label='With buffer (Surveillance)', color='orange', marker='o', linestyle='-')
    ax.plot(alpha, improved_emergency_list, label='With buffer', color='orange', marker='v', linestyle='-')
    ax.plot(alpha, emergency_list, label='Without buffer', color='#824D1B', marker='v', linestyle='--')
    ax.set_title(dataset, fontsize=FONTSIZE - 4)
    ax.set_ylim(np.min([improved_emergency_list, emergency_list]) * 0.9,
                np.max([improved_emergency_list, emergency_list]) * 1.1)
plot_postprocess(ax, fig, 'Effectiveness of dynamically weighted buffer',
                 '$I_{\\text{emer}}$')