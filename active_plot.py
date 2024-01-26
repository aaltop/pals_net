def active_plotting(axes, x_y_data):
    '''
    Change a plot interactively. This assumes that <axes> have
    been plotted to once, and `plot.show(block=False)` has been called.

    Parameters
    ----------

    ### axes : iterable of matplotlib Axes
    
    ### x_y_data : list of tuples of (x,y)
        Should be as long as <axes>, with each x and y representing
        data to be plotted to the corresponding value in <axes>.
    '''

    for i in range(len(axes)):

        ax = axes[i]
        x,y = x_y_data[i]
        line, = ax.get_lines()
        line.set_data(x,y)
        ax.relim()
        ax.autoscale(True)


if __name__ == "__main__":

    pass