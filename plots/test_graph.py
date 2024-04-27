from unittest import TestCase
from graph import compare_plot


class Test(TestCase):
    def test_compare_plot(self):
        compare_plot(x_label="No. of PoIs",
                     y_label="Energy efficiency",
                     x=[100, 200, 300, 400, 500],
                     yrange=[0, 0.45],
                     data_dict=dict(Ours=[0.3, 0.452, 0.478, 0.506, 0.6023],
                                    DPPO=[0.268, 0.392, 0.46, 0.453, 0.485],
                                    EDICS=[0.13, 0.2, 0.25, 0.283, 0.2165],
                                    DC=[0.097, 0.1049, 0.2947, 0.3137, 0.2787],
                                    Greedy=[0.0646, 0.2324, 0.2, 0.2974, 0.1278]),
                     group_name="test_graph"
                     )
