'''
Functions used for processing input and output data
'''

import logging
import abc

from read_data import (
    read_metadata, 
    get_train_or_test,
    get_components,
)

import torch

from pytorch_helpers import (
    convert_to_tensor
)


# TODO: maybe just replace this with a convolution?
def process_input_mlp(data, num_of_channels=None, take_average_over=None, start_index=None):
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
    process_input_mlp(inputs, num_of_channels, take_average_over=take_average_over, start_index=start_index)

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

class AbstractOutputProcessing(abc.ABC):
    '''
    Base Abstract class for output processing
    '''

    @abc.abstractmethod
    def process(self, _input, *args, **kwargs):
        '''
        Process the input and return the result.
        '''
        pass

    @abc.abstractmethod
    def inv_process(self, _input, *args, **kwargs):
        '''
        The inverse of `process()`.
        '''
        pass

    @abc.abstractmethod
    def inv_process_var(self, _input, *args, **kwargs):
        '''
        The inverse processing of variance of the output.
        '''
        pass

    @abc.abstractmethod
    def state_dict(self) -> dict:
        '''
        Return the attributes necessary for processing, such
        that the state of the instance can be instantiated again
        by passing the return from this method to the __init__
        of the class: OutputProcessing(**output_processing.get_state()).
        '''
        pass

class DivByMax(AbstractOutputProcessing):
    '''
    Divide <output> by <train_output_col_max>, 
    the maximum of each column of the training data.
    '''

    def __init__(self, train_output=None, train_output_col_max=None, no_processing_idx=None):
        '''
        Calculate <train_output_col_max> from <train_output> if
        <train_output_col_max> is None. If <no_process_idx> is given,
        the indices given by it will not be processed.
        '''

        if not (train_output_col_max is None):
            self.train_output_col_max = train_output_col_max
        elif not (train_output is None):
            self.train_output_col_max = train_output.amax(dim=0)
        else:
            raise TypeError("One of <train_output>, <train_output_col_max> should be a suitable PyTorch tensor.")
        
        max_col = self.train_output_col_max
        # set any zero max values to one for the division
        zeros = torch.zeros(size=[len(max_col)]).to(max_col.device)
        self.train_output_col_max[torch.isclose(max_col, zeros)] = 1.0
        
        if not (no_processing_idx is None):
            self.no_processing_idx = no_processing_idx
            self.train_output_col_max[no_processing_idx] = 1.0
            
    
    def process(self, _input):

        return _input/self.train_output_col_max
    
    def inv_process(self, _input):

        return _input*self.train_output_col_max
    
    def inv_process_var(self, _input):
        '''
        The inverse processing of variance of the output.
        '''
        return _input*self.train_output_col_max**2

    def state_dict(self):

        return {
            "train_output_col_max":self.train_output_col_max,
            "no_processing_idx":self.no_processing_idx
        }
    

class SubMinDivByMax(AbstractOutputProcessing):
    '''
    Subtract <train_output_col_min>, the minimum of each column of the training data from <output>,
    and divide by <train_output_col_max>, the maximum of each column of the training data. The max
    is calculated after the subtraction. 
    '''

    def __init__(self, train_output=None, train_output_col_min=None, train_output_col_max=None, no_processing_idx=None):
        '''
        Calculate <train_output_col_min> and <train_output_col_max> from <train_output>
        based on whether they are None or not. If <no_process_idx> is given,
        the indices given by it will not be processed.
        '''

        if not (train_output_col_min is None):
            self.train_output_col_min = train_output_col_min
        elif not (train_output is None):
            self.train_output_col_min = train_output.amin(dim=0)
        else:
            raise TypeError("One of <train_output>, <train_output_col_min> should be a suitable PyTorch tensor.")
        
        if not (train_output_col_max is None):
            self.train_output_col_max = train_output_col_max
        elif not (train_output is None):
            self.train_output_col_max = (train_output-self.train_output_col_min).amax(dim=0)
        else:
            raise TypeError("One of <train_output>, <train_output_col_max> should be a suitable PyTorch tensor.")
        
        max_col = self.train_output_col_max
        # set any zero max values to one for the division
        zeros = torch.zeros(size=[len(max_col)]).to(max_col.device)
        self.train_output_col_max[torch.isclose(max_col, zeros)] = 1.0
        
        if not (no_processing_idx is None):
            self.no_processing_idx = no_processing_idx
            self.train_output_col_max[no_processing_idx] = 1.0
            self.train_output_col_min[no_processing_idx] = 0.0
    
    def process(self, _input):

        return (_input-self.train_output_col_min)/self.train_output_col_max
    
    def inv_process(self, _input):

        return _input*self.train_output_col_max+self.train_output_col_min
    
    def inv_process_var(self, _input):
        '''
        The inverse processing of variance of the output.
        '''
        return _input*self.train_output_col_max**2
    
    def state_dict(self):

        return {
            "train_output_col_max":self.train_output_col_max,
            "train_output_col_min":self.train_output_col_min,
            "no_processing_idx":self.no_processing_idx
        }

def process_input01(x):
    '''
    Assume x is input data, with rows being data points.
    '''

    x += 1
    x = x/torch.amax(x, dim=1).reshape((-1,1))
    x = torch.log(x)
    return x
    