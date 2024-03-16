import os

from helpers import date_time_str

import matplotlib.pyplot as plt


class PlotSaver:


    def __init__(self, file_suffix=None):

        folder_name = date_time_str() + file_suffix if not (file_suffix is None) else ""
        self.figs_path = os.path.join(os.getcwd(), "figures", folder_name)
        os.makedirs(self.figs_path, exist_ok=True)

    def get_saved_path(self, name):

        return os.path.join(self.figs_path, name.replace(" ", "_")+".png")

    def save(self, name, fig=None):

        if fig is None:
            fig = plt.gcf()
        cur_size = fig.get_size_inches()
        fig.set_size_inches((19.2,10.8))
        save_path = self.get_saved_path(name)
        fig.savefig(save_path)
        fig.set_size_inches(cur_size)
        return save_path

    