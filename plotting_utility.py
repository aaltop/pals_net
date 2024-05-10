import os

from helpers import date_time_str

import matplotlib.pyplot as plt


class PlotSaver:


    def __init__(self, file_suffix=None, save_resolution=(19.2,10.8)):

        folder_name = date_time_str() + file_suffix if not (file_suffix is None) else ""
        self.figs_path = os.path.join(os.getcwd(), "figures", folder_name)

        self.resolution = save_resolution

    def get_saved_path(self, name):

        return os.path.join(self.figs_path, name.replace(" ", "_")+".png")

    def save(self, name, fig=None):

        os.makedirs(self.figs_path, exist_ok=True)

        if fig is None:
            fig = plt.gcf()
        cur_size = fig.get_size_inches()
        fig.set_size_inches(self.resolution)
        save_path = self.get_saved_path(name)
        fig.savefig(save_path)
        fig.set_size_inches(cur_size)
        return save_path

    