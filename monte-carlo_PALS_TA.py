# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:00:49 2023.

@author: René Bes
"""

import numpy as np
import pandas as pd
import json
import os


def detector_response(event_times, sigma, rng):
    """
    Broaden detection time by detector's time response.

    Parameters
    ----------
    event_times : numpy array
        Photon travel time (averaged value) from emmission point to detector
        to be convoluted by detector time response, in ps. Detector time
        response is taken as random gaussian broadening
        as detector size/geometrical effect.
    sigma : float
        Detector's time response, in ps.

    Returns
    -------
    None.

    """
    size = np.size(event_times)
    event_times += rng.normal(0, sigma, size)


def sim_bkg(num_counts, time_gate, rng):
    """
    Simulate the randomly distributed background, aka random false coincidence

    Parameters
    ----------
    num_counts : int
        Number of events to generate.
    time_gate : float
        Time gate used for the coincidence, in ps.

    Returns
    -------
    bkg_events : numpy array
        Generated array of background events.

    """
    bkg_events = rng.uniform(0, time_gate, num_counts)

    return bkg_events


def sim_coinc(num_counts, lifetime, sigma_start, sigma_stop, offset, rng):
    """
    Simulate the coincidence events for a given positron component.

    Parameters
    ----------
    num_counts : int
        Number of events to generate.
    lifetime : float
        Component lifetime, in ps.
    sigma_start : float
        Time response (sigma) of start detector, in ps.
    sigma_stop : float
        Time response (sigma) of stop detector, in ps.
    offset : float
        Time offset of the detection system, in ps.

    Returns
    -------
    events : numpy array
        Generated array of coincidence events.

    """
    starts = np.zeros(num_counts)
    # Broaden detection time by detector's time response
    detector_response(starts, sigma_start, rng)

    stops = rng.exponential(lifetime, num_counts)
    # # Broaden detection time by detector's time response
    detector_response(stops, sigma_stop, rng)
    events = stops - starts + offset

    return events


def do_histogram(events, time_gate, bin_size):
    """
    Sort and histogram the generated events.

    Parameters
    ----------
    events : numpy array
        Array of generated events.
    time_gate : dict
        Bin size and time gate used for coincidence, both in ps.

    Returns
    -------
    hist : numpy array
        Histogram of sorted events, in counts.
    bins : numpy array
        Binning of histogram, in ps.

    """
    num_bin = round(time_gate/ bin_size)
    # bins = np.linspace(0, time_gate, num_bin)
    hist, bins = np.histogram(events, bins=num_bin, range=(0, time_gate))
    return hist, bins[:-1]


def sim_pals_separate(input_prm, rng):
    """
    Simulate the coincidence events from input parameters,
    return events separately for each component.

    Parameters
    ----------
    input_prm : dict
        Input parameters which includes:
            "num_events": total number of events to generate,
            "bkg": relative intensity of the random background events,
            "components": list of lifetime components, consisting in a tuple of
            lifetime value (in ps) and its relative intensity,
            "bin_size": size of the binnin for the coincidenec histogram,
            i.e. the actual PALS spectrum,
            "time_gate": maximum time where coincidence are considered, in ps,
            "sigma_start": start detector time response (sigma),
            "sigma_stop": stop detector time response (sigma),
            "offset": time zero delay of the detection system.

    Returns
    -------
    separate_events : dict of numpy arrays
        The events times are for the background and each component.

        events keys are "component <n>" for each component, where <n> is the
        number of the component in input parameters, starting from zero (0).
        The background radiation has the key "background".

    """
    num_events = input_prm["num_events"]
    bkg = input_prm["bkg"]
    components = input_prm["components"]
    time_gate = input_prm["time_gate"]
    sigma_start = input_prm["sigma_start"]
    sigma_stop = input_prm["sigma_stop"]
    offset = input_prm["offset"]


    # background events calculations
    num_counts = int(num_events * bkg)
    bkg_events = sim_bkg(num_counts, time_gate, rng)

    separate_events = {}
    # component related events calculations
    for idx, comp in enumerate(components):
        lifetime, component_intensity = comp
        num_counts = int(num_events * component_intensity)
        comp_events = sim_coinc(num_counts, lifetime, sigma_start,
                                sigma_stop, offset, rng)


        separate_events[f"component {idx}"] = comp_events

    separate_events["background"] = bkg_events


    return separate_events

def sim_pals(input_prm, rng):
    """
    Simulate the coincidence events from input parameters.

    Parameters
    ----------
    input_prm : dict
        Input parameters which includes:
            "num_events": total number of events to generate,
            "bkg": relative intensity of the random background events,
            "components": list of lifetime components, consisting in a tuple of
            lifetime value (in ps) and its relative intensity,
            "bin_size": size of the binnin for the coincidenec histogram,
            i.e. the actual PALS spectrum,
            "time_gate": maximum time where coincidence are considered, in ps,
            "sigma_start": start detector time response (sigma),
            "sigma_stop": stop detector time response (sigma),
            "offset": time zero delay of the detection system.

    Returns
    -------
    bins : numpy array
        Binning of histogram, in ps.
    total_hist : numpy array
        Histogram of the generated and sorted events, in counts.

    """
    num_events = input_prm["num_events"]
    bkg = input_prm["bkg"]
    components = input_prm["components"]
    bin_size = input_prm["bin_size"]
    time_gate = input_prm["time_gate"]
    sigma_start = input_prm["sigma_start"]
    sigma_stop = input_prm["sigma_stop"]
    offset = input_prm["offset"]


    # background events calculations
    num_counts = int(num_events * bkg)
    bkg_events = sim_bkg(num_counts, time_gate, rng)

    # component related events calculations
    for idx, comp in enumerate(components):
        lifetime, component_intensity = comp
        num_counts = int(num_events * component_intensity)
        comp_events = sim_coinc(num_counts, lifetime, sigma_start,
                                sigma_stop, offset, rng)

        if idx == 0:
            coinc_events = comp_events

        else:
            coinc_events = np.concatenate((coinc_events, comp_events))


    total_events = np.concatenate((bkg_events, coinc_events))

    # Sorting and histogramming of all events
    total_hist, bins = do_histogram(total_events, time_gate, bin_size)

    return bins, total_hist

def concat_events(separate_events:list):
    '''
    Concatenate all events.

    Parameters
    ----------

    separate_events :
        a list of all events.
    '''

    total_events = separate_events[0].copy()

    for event in separate_events[1:]:
        total_events = np.concatenate((total_events, event))

    return total_events

def write_sim_metadata(
        input_prm:dict,
        file_name:str,
        folder_path:str,
):
    '''
    Write metadata (input parameters) for simulated
    data into file called "metadata.txt".
    '''


    file_path = os.path.join(folder_path, "metadata.txt")

    # create file if necessary
    if not os.path.isfile(file_path):
        # encoding is not needed, but pylint
        # will complain
        with open(file_path, "x", encoding="utf-8"):
            pass

    # do read and write separately so as to avoid
    # dealing with anything extra. This is perhaps
    # a little more clear, even if not as fast as
    # another approach.
    with open(file_path, "r+", encoding="utf-8") as rf:
        # initialise as empty dict if file is empty
        if (os.stat(file_path).st_size == 0):
            rf.write("{}")
            rf.seek(0)

        all_metadata = json.load(rf)
    all_metadata[file_name] = input_prm

    with open(file_path, "w", encoding="utf-8") as wf:
        json.dump(all_metadata, wf, indent=4)


def write_simulation_data(
    input_prm:dict,
    folder_name:str=None,
    file_name_beginning:str=None,
) -> None:
    '''
    Write simulation data to a file.

    Pararameters
    ------------

    input_prm : dict
        Input parameters for function sim_pals.
    folder_name : str, default None
        Name of the folder to write data into.
        If not specified, defaults to "simdata"
        in current working directory.
    file_name_beginning : str, default None
        Name of file to write data into. Final file
        will be "<file_name_beginning>_<num>.pals", where
        <num> is automatically determined based on how many files
        there already are.

    Returns
    -------

    None
        
    '''

    # determine the folder to write into,
    # create it if need be
    folder_name = "simdata"
    cwd = os.getcwd()
    folder_path = os.path.join(cwd,folder_name)

    if not (os.path.isdir(folder_path)):
        os.mkdir(folder_path)

    # automatically determine the file name
    file_index = 1
    file_name_beginning = "simdata"
    file_name = "{0}_{1:05}.pals"
    while True:
        final_file_name = file_name.format(file_name_beginning, file_index)
        file_path = os.path.join(
            folder_path, final_file_name)
        
        if os.path.isfile(file_path):
            file_index += 1
            continue
        break

    # generate simulation data, set in
    # dataframe
    rng = np.random.default_rng()
    time_ps, counts = sim_pals(input_prm, rng)
    pd_data = {"time_ps":time_ps, "counts":counts}
    df = pd.DataFrame(data=pd_data, dtype=np.float64)

    with open(file_path, "w", encoding="utf-8") as f:
        df.to_csv(
            path_or_buf=f,
            sep = " ",
            float_format="{:.18e}".format,
            index=False,
            line_terminator="\n"
        )

    # Don't want to have simdata without
    # the corresponding metatdata. Would
    # be much easier to just write metadata
    # in the same file as the data, though.
    try:
        write_sim_metadata(
            input_prm,
            final_file_name,
            folder_path)
    except BaseException as e:
        # maybe should write to stderr?
        print("\nException occured while writing metadata. Deleting simdata file.\n")
        os.remove(file_path)
        raise e


def plot_sim_data():
    import matplotlib.pyplot as plt

    input_prm = {"num_events": 1_000_000,
                 "bkg": 0.05,
                 "components": [(415, .10), (232, .50), (256, .30), (1200, .05)],
                 "bin_size": 25,
                 "time_gate": 15_000,
                 "sigma_start": 68,
                 "sigma_stop": 68,
                 "offset": 2000}

    rng = np.random.default_rng() # Seed to be changed for repeated simulation

    # gets events separately so can also look at 
    # the separate components in addition to
    # the overall spectrum
    separate_events = sim_pals_separate(input_prm, rng)

    total_events = concat_events(list(separate_events.values()))

    total_hist, bins = do_histogram(total_events, input_prm["time_gate"], input_prm["bin_size"])

    plt.plot(bins,total_hist, label="total")

    print(total_hist, bins)

    for name,separate in separate_events.items():

        sep_hist, sep_bins = do_histogram(separate, input_prm["time_gate"], input_prm["bin_size"])
        plt.plot(sep_bins, sep_hist, label=name)


    # input_prm["offset"] = 1000
    # bins2, total_hist2 = sim_pals(input_prm, rng)
    # plt.plot(bins2, total_hist2, "r")

    plt.legend()
    plt.yscale("log")
    print(total_hist.mean())
    plt.show()

def main():

    input_prm = {"num_events": 1_000_000,
                 "bkg": 0.05,
                 "components": [(415, .10), (232, .50), (256, .30), (1200, .05)],
                 "bin_size": 25,
                 "time_gate": 15_000,
                 "sigma_start": 68,
                 "sigma_stop": 68,
                 "offset": 2000}

    for _ in range(100):

        write_simulation_data(input_prm)

if __name__ == "__main__":

    main()
