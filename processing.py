'''
Functions used for processing input and output data
'''

import logging

from read_data import (
    read_metadata, 
    get_train_or_test,
    get_components,
)

from pytorch_helpers import (
    convert_to_tensor
)


def process_input(data, num_of_channels=None, take_average_over=None, start_index=None):
    '''
    Cut off counts in <data> such that only data from the
    <start_index> onwards up to <num_of_channels> beyond the <start_index> 
    is considered, and takes non-rolling means of <take_average_over> values 
    of the result. Finally, normalises the resulting averaged data.

    Parameters
    ----------
    ### data : list of numpy arrays
        The counts for <data.size> simulated spectra.

    ### num_of_channels : int
        How many channels to take beyond the max channel,
        as in [max_chan:max_chan+num_of_channels]. Preferably
        divisible by <take_average_over>. Default is 100.

    ### take_average_over : int
        how many channels to average over. Default is 5.

    ### start_index : int
        The first index to include in the end result. If None,
        takes the index of the max value.

    
    Returns
    -------
        None, the operation is performed in-place.
    '''

    if num_of_channels is None:
        num_of_channels = 100
    if take_average_over is None:
        take_average_over = 5

    for i in range(len(data)):
        if start_index is None:
            start_index = data[i].argmax()
        # average over <take_average_over> value groups (non-rolling)
        used_data = data[i][start_index:start_index+num_of_channels]
        # take a divisible number of data points
        data_length = (used_data.flatten().shape[0]//take_average_over)*take_average_over
        
        averaged_data = used_data[:data_length].reshape((-1,take_average_over)).mean(axis=1)
        # normalise
        data[i] = averaged_data/averaged_data.max()

def fetch_and_process_input(
        folder_path, 
        data_files, 
        take_average_over=None,
        start_index=None
        ):
    '''
    Fetch and process train/test input data.

    Parameters
    ----------

    ### data_files : list of strings
        The names of the files which contain the input data. Should
        be a list also for just one file.


    ### take_average_over : int
        In processing the input, a non-rolling average is calculated.
        This determines how many data points are included in each 
        averaging.
        


    Returns
    -------

    ### inputs : PyTorch tensor
        The input data.


    '''

    inputs = get_train_or_test(folder_path, data_files)

    # not necessary sensible, but ensures that all channels beyond
    # start_index are taken (could of course be changed)
    num_of_channels = inputs[0].shape[0]

    if take_average_over is None:
        take_average_over = 5
    
    if start_index is None:
        start_index = 0

    logging.info(f"averaging input over {take_average_over} bins")
    process_input(inputs, num_of_channels, take_average_over=take_average_over, start_index=start_index)

    return convert_to_tensor(inputs)

def fetch_output(folder_path, data_files):
    '''
    Fetch output (components). For use with older type of
    data saving, where simulation input parameters are kept in metadata
    file. The metadata file consists of a json-formatted "dictionary",
    with keys being the data files' names, and the values for these
    being the input parameter dictionaries. 

    Returns
    -------

    ### outputs : PyTorch tensor
        The components that created the simulation data.
    '''

    
    metadata = read_metadata(folder_path)
    outputs = get_components(metadata, data_files)
    return convert_to_tensor(outputs)