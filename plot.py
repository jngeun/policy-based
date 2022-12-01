import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class LearningPlotter():
    def __init__(self, dir_path=""):
        self.dir_path = Path(dir_path)
        self.file_list = sorted(self.dir_path.glob('*.npy'))

        sns.set(rc={'figure.figsize':(10, 10)})
        # seaborn style
        # https://seaborn.pydata.org/generated/seaborn.axes_style.html#seaborn.axes_style
        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": "-"})
        # sns.set_style("whitegrid", {'axes.grid' : True, 'axes.edgecolor':'black'})


    def plot(self, file, label=None, max_idx=None):
        file = Path(file)
        data = np.load(file, 'r')
        df = pd.DataFrame({"Episode":np.arange(1, data.shape[0]+1, dtype=int), file.stem:data})

        sns_plot = sns.lineplot(x="Episode", y=file.stem, data=df, label=label)
        fig = sns_plot.get_figure()

        return fig

    def compare_algorithm(self, file1, file2, save_location):
        self.plot(file1, label="REINFORCE")
        fig = self.plot(file2, label="REINFORCE_with_baseline")

        fig.savefig(save_location)

if __name__ == "__main__":
    file_path1 = "/home/user/policy-based/results/CartPole/REINFORCE/scores.npy"
    file_path2 = "/home/user/policy-based/results/CartPole/REINFORCE_with_Baseline/scores.npy"
    save_location = "/home/user/policy-based/figure/reinforce_vs_reinforce_with_baseline.png"

    learning_plotter = LearningPlotter()
    learning_plotter.compare_algorithm(file_path1, file_path2, save_location)