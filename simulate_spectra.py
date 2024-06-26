# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:00:49 2023.

@author: René Bes, Topi Aaltonen
"""
import json
import os
import numpy as np

_rng = np.random.default_rng()


def detector_response(event_times, sigma, rng=None):
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

    if rng is None:
        rng = _rng

    size = np.size(event_times)
    event_times += rng.normal(0, sigma, size)


def sim_bkg(num_counts, time_gate, rng=None):
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

    if rng is None:
        rng = _rng

    bkg_events = rng.uniform(0, time_gate, num_counts)

    return bkg_events


def sim_coinc(num_counts, lifetime, sigma_start, sigma_stop, offset, rng=None):
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

    if rng is None:
        rng = _rng

    starts = np.zeros(num_counts)
    # Broaden detection time by detector's time response
    if  sigma_start > 0:
        detector_response(starts, sigma_start, rng)

    stops = rng.exponential(lifetime, num_counts)
    # # Broaden detection time by detector's time response
    if sigma_stop > 0:
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


def sim_pals_separate(input_prm, rng=None):
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

    if rng is None:
        rng = _rng

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

def sim_pals(input_prm, rng=None):
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

    if rng is None:
        rng = _rng

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
    file_index:int=None,
    rng=None
) -> int:
    '''
    Write simulation data and metadata to a file. Writes <input_prm>
    on the first line of the determined file, and writes simulated
    spectrum data (time in picoseconds, counts) after this. Should be
    then possible to read the contents by deserialising the first line
    with json and reading the rest with np.loadtxt.

    Pararameters
    ------------

    ### input_prm : dict
        Input parameters for function sim_pals.

    ### folder_name : str, default None
        Name of the folder to write data into.
        If not specified, defaults to "simdata"
        in current working directory.

    ### file_name_beginning : str, default None
        Name of file to write data into. Final file
        will be "<file_name_beginning>_<num>.pals", where
        <num> is either <file_index> or any next available index
        with zero padding. 
        
        If None, defaults to "simdata".

    ### file_index : int, default None
        The number of the file to try and write the data into. If
        the corresponding file is already used, will try the next
        files (following indices) until available or max index is reached.

        If None, will try each index in order starting from 1.

    Returns
    -------

    ### file_index : int
        the number of the file that was written to
        
    '''

    if rng is None:
        rng = _rng

    # determine the folder to write into,
    # create it if need be

    if folder_name is None:
        folder_name = "simdata"
    cwd = os.getcwd()
    folder_path = os.path.join(cwd,folder_name)

    if not (os.path.isdir(folder_path)):
        os.mkdir(folder_path)

    # automatically determine the file name (could probably use a
    # bisect, but might not have that big an effect. Again, point
    # is to write a lot of files at a time, and the index only needs
    # to be determined once in this case.)
    # -----------------------------------------------------------------
    if file_index is None:
        file_index = 1

    if file_name_beginning is None:
        file_name_beginning = "simdata"
    
    file_name = "{0}_{1:05}.pals"
    while True:
        final_file_name = file_name.format(file_name_beginning, file_index)
        file_path = os.path.join(
            folder_path, final_file_name)
        
        if os.path.isfile(file_path):
            file_index += 1
            if file_index >= 100_000:
                raise ValueError(f"file_index too high")
            continue
        break
    # ===============================================================

    # generate simulation data
    time_ps, counts = sim_pals(input_prm, rng)
    data = np.stack((time_ps, counts), axis=-1)

    # write simulation data and metadata (could be own function)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(input_prm, f)
            f.write("\n")
            np.savetxt(f, data, fmt=r"%d")

    # don't want incomplete data files
    except BaseException as e:
        # maybe should write to stderr?
        print("\nException occured while writing data. Deleting data file.\n")
        os.remove(file_path)
        raise e
    
    # return file index so next write can more easily determine file
    # to write to
    return file_index


def plot_sim_data(input_prm=None, rng=None):
    import matplotlib.pyplot as plt

    if rng is None:
        rng = _rng

    if input_prm is None:
        input_prm = {"num_events": 1_000_000,
                    "bkg": 0.05,
                    "components": [(415, .10), (232, .50), (256, .30), (1200, .05)],
                    "bin_size": 25,
                    "time_gate": 15_000,
                    "sigma_start": 68,
                    "sigma_stop": 68,
                    "offset": 2000}

    # gets events separately so can also look at 
    # the separate components in addition to
    # the overall spectrum
    separate_events = sim_pals_separate(input_prm, rng)

    total_events = concat_events(list(separate_events.values()))

    total_hist, bins = do_histogram(total_events, input_prm["time_gate"], input_prm["bin_size"])

    plt.plot(bins,total_hist, label="total")

    for name,separate in separate_events.items():

        sep_hist, sep_bins = do_histogram(separate, input_prm["time_gate"], input_prm["bin_size"])
        plt.plot(sep_bins, sep_hist, label=name)


    # input_prm["offset"] = 1000
    # bins2, total_hist2 = sim_pals(input_prm, rng)
    # plt.plot(bins2, total_hist2, "r")

    plt.legend()
    plt.xlabel("time")
    plt.ylabel("counts")
    plt.yscale("log")
    plt.show()


def write_many_simulations(sims_to_write, input_prm, folder_name=None, repetition_count=1):
    '''
    Write multiple simulations to file; see function
    `write_simulation_data()` for more information.

    Parameters
    ----------

    ### sims_to_write : int
        Number of simulations to write to file.

    ### input_prm : callable
        Returns the parameters used for the simulation, for example

        input_prm = {"num_events": 1_000_000,
                    "bkg": 0.05,
                    "components": [(415, .10), (232, .50), (256, .30), (1200, .05)],
                    "bin_size": 25,
                    "time_gate": 15_000,
                    "sigma_start": 68,
                    "sigma_stop": 68,
                    "offset": 2000}

    ### folder_name : string
        Folder to write simulations into. See function
        `write_simulation_data()` for more information.

    ### repetition_count : int
        How many times to use the same input_prm, mainly useful for
        when using `random_input_prm()`. Final number of simulations
        will be repetition_count*sims_to_write.

    Returns
    -------

        None
    '''

    file_index = None

    folder_path = os.path.join(os.getcwd(), folder_name)

    os.makedirs(folder_path, exist_ok=True)

    for i in range(sims_to_write):
        print("\033[K",end="")
        print(f"{i+1}/{sims_to_write}", end="\r")
            
        prm = input_prm()

        for _ in range(repetition_count):
            file_index = write_simulation_data(
                prm, 
                folder_name=folder_name,
                file_index=file_index
            )
            
            # write_simulation_data should return previously written
            # file index, so add one to get next free one
            file_index += 1

    print()