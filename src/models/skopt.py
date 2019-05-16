import matplotlib.pyplot as plt

from skopt.plots import plot_convergence
from IPython.display import clear_output


class ConvergencePlotCallback(object):
    def __init__(self, figsize=(12, 8)):
        self.fig = plt.figure(figsize=figsize)

    def __call__(self, res):
        clear_output(wait=True)
        plot_convergence(res)
        plt.show()
