import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class LearningPlotter():
    def __init__(self, dir_path=""):
        self.dir_path = Path(dir_path)
        self.file_list = sorted(self.dir_path.glob('*.npy'))


    def plot(self, file):
        file = Path(file)
        data = np.load(file, 'r')
        df = pd.DataFrame({"Episode":np.arange(1, data.shape[0]+1, dtype=int), file.stem:data})
        sns.set_style("darkgrid", {'axes.grid' : True, 'axes.edgecolor':'black'})
        sns_plot = sns.lineplot(x="Episode", y=file.stem, data=df)
        fig = sns_plot.get_figure()
        fig.savefig(file.with_suffix('.png'))


if __name__ == "__main__":
    file_path = "/home/user/policy-based/results/Pendulum/REINFORCE/scores.npy"

    learning_plotter = LearningPlotter()
    learning_plotter.plot(file_path)