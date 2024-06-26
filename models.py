import abc
from collections import namedtuple

from pytorch_helpers import pretty_print, conv_expand, conv_compress

import torch

class AbstractModel(abc.ABC):
    '''
    Used to define a common interface for Pytorch-based models. The
    motivation is mainly how loading in a model from its state_dict
    works, which seems to require (looking at the PyTorch documentation) 
    having the model instantiated first.
    For this, it's perhaps simplest to also save the parameters used
    in instantiating the model object, so that when loading the state,
    instantiation can be done simply by passing the kwarg dict to the
    class. This could possibly be helped by using TorchScript, but
    I'd rather not spend time on ironing out the kinks with that right now.

    A second reason for this class is having a common interface for
    pretty printing the model.
    '''

    @abc.abstractproperty
    def instantiation_kwargs(self):
        '''
        Returns the arguments used in instantiating the object as a
        kwarg dict.
        '''

        return self._instantiation_kwargs
    
    def __str__(self):

        return pretty_print(self, self.instantiation_kwargs)

class MLP(torch.nn.Module, AbstractModel):
    '''

    A Multi-Layer Perceptron.

    Parameters
    ----------


    layer_sizes : list of ints
        first should be input size, last
        shoud be output size, in between are
        the sizes of the hidden layers.
    '''

    def __init__(self, layer_sizes):

        super().__init__()

        self._instantiation_kwargs = {
            "layer_sizes": layer_sizes
        }

        self.layers = torch.nn.ModuleList()

        # for easy access to layer sizes outside the class
        self.layer_sizes = layer_sizes

        for i in range(len(layer_sizes)-1):
            inf = layer_sizes[i]
            outf = layer_sizes[i+1]
            self.layers.append(
                torch.nn.Linear(in_features=inf, out_features=outf)
            )


    def forward(self, x):

        for layer in self.layers[:-1]:

            x = torch.nn.functional.relu(layer(x))

        x = self.layers[-1](x)

        return x
    
    @property
    def instantiation_kwargs(self):

        return super().instantiation_kwargs

class Conv1(torch.nn.Conv1d):
    '''
    Modifies the PyTorch Conv1d to be usable with input data of the
    form (batch_size, signal_length).
    '''

    def forward(self, input):

        input = conv_expand(input)
        return super().forward(input)

class NeuralNet(torch.nn.Module, AbstractModel):
    '''
    For generic networks.
    '''

    def __init__(self, layers):
        '''
        

        Parameters
        ----------

        ### layers : list of (module, boolean)
            The module should subclass torch.nn.Module, and the boolean
            determines whether an activation function should be applied
            to that layer.

        '''

        super().__init__()

        self._instantiation_kwargs = {
            "layers": layers
        }

        modules, use_activation = zip(*layers)

        self.layers = torch.nn.ModuleList(modules)
        
        activation_function = torch.nn.functional.relu
        # for whether to use activation function or just do identity
        self.activation = [activation_function if acti else lambda val: val for acti in use_activation]

    def forward(self, x):

        for i, layer in enumerate(self.layers[:-1]):
                x = self.activation[i](layer(x))

        x = self.layers[-1](x)

        return x

    @property
    def instantiation_kwargs(self):

        return super().instantiation_kwargs

PALS_output = namedtuple("PALS_output", ("normal", "softmax"))
class PALS_MSE(NeuralNet):
    '''
    At the end, computes a softmax on some of the input from the previous
    layer.

    Used with Mean Squared Error.
    '''


    def __init__(self, layers, idx):
        '''
        
        Parameters
        ----------

        ### layers : list of (module, boolean)
            The module should subclass torch.nn.Module, and the boolean
            determines whether an activation function should be applied
            to that layer.


        ### idx : list of list of int
            Element one contains a list of indices to not compute
            softmax on, while element two contains a list of indices
            to compute softmax on.

        '''

        super().__init__(layers)
        self._instantiation_kwargs["idx"] = idx

        # assume data points in rows
        self.softmax = torch.nn.Softmax(1)
        self.idx = idx

    def process_output(self, x):
        '''
        For processing the output from the last layer.
        '''

        normal_idx, softmax_idx = self.idx

        # technically unnecessary, could just take the right amount
        # in sequence
        normal = x[:,normal_idx]
        softmaxxed = self.softmax(x[:,softmax_idx])
        return PALS_output(normal, softmaxxed)

    def forward(self, x):

        x = super().forward(x)

        return self.process_output(x)
    
    @property
    def instantiation_kwargs(self):

        instantiation_kwargs = super().instantiation_kwargs
        return instantiation_kwargs

MeanAndVar = namedtuple("MeanAndVar", ("mean", "var"))
class PALS_GNLL(PALS_MSE):
    '''
    At the end, computes a softmax on some of the input from the previous
    layer. Also returns variance for predictions.

    Used with Gaussian Negative Log-Likelihood Loss.
    '''

    def __init__(self, layers, idx):
        '''
        
        Parameters
        ----------

        ### layers : list of (module, boolean)
            The module should subclass torch.nn.Module, and the boolean
            determines whether an activation function should be applied
            to that layer.


        ### idx : list of list of int
            Element one contains a list of indices of values to not
            compute softmax on, 
            element two contains a list of indices for the matching variances,
            element three contains a list of indices to compute softmax on, and element four contains a list
            of indices for the matching variances.

        '''
        super().__init__(layers, idx)


    def process_output(self, x):

        normal_idx, normal_var_idx, softmax_idx, softmax_var_idx = self.idx

        softplus = torch.nn.Softplus(beta=10)

        normal = x[:, normal_idx]
        normal_var = softplus(x[:, normal_var_idx])
        softmaxxed = self.softmax(x[:, softmax_idx])
        softmaxxed_var = softplus(x[:, softmax_var_idx])

        output = PALS_output(
            MeanAndVar(normal, normal_var), 
            MeanAndVar(softmaxxed, softmaxxed_var)
        )
        
        return output
        

    def forward(self, x):
        '''
        Returns
        -------

        ### output : PALS_output(MeanAndVar, MeanAndVar)
            Contains the normal and softmax outputs, both of
            which contain the predicted mean and predicted variance.
        
        '''

        return super().forward(x)


class PALS_GNLL_Intensities(PALS_GNLL):

    def process_output(self, x):
        
        softmax_idx, var_idx = self.idx

        softplus = torch.nn.Softplus(beta=10)

        sigm = torch.nn.Sigmoid()

        # softmaxxed = self.softmax(x[:, softmax_idx])
        # softmaxxed = sigm(x[:, softmax_idx])
        softmaxxed = x[:, softmax_idx]
        softmaxxed_var = softplus(x[:, var_idx])
        
        return [softmaxxed, softmaxxed_var]

class PALS_GNLL_Single(PALS_GNLL):

    def process_output(self, x):
        normal, var = self.idx
        softplus = torch.nn.Softplus(beta=10)
        return [x[:,normal], softplus(x[:,var])]

if __name__ == "__main__":

    layers = [5,3,2]

    model = MLP(layers)
    print(model.layer_sizes)
    print(model)
    model2 = MLP(**model.instantiation_kwargs)
    print(model2)

    # linear = torch.nn.LazyLinear
    # conv = torch.nn.Conv1d
    # nn = NeuralNet(
    #     conv(1,3,5,5),
    #     conv(1,1,3,),
    #     linear()
    # )