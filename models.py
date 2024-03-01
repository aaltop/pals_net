import abc

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

    def forward(self, x):

        x = super().forward(x)

        normal_idx, softmax_idx = self.idx

        # technically unnecessary, could just take the right amount
        # in sequence
        normal = x[:,normal_idx]
        softmaxxed = self.softmax(x[:,softmax_idx])
        return normal, softmaxxed 
    
    @property
    def instantiation_kwargs(self):

        instantiation_kwargs = super().instantiation_kwargs
        return instantiation_kwargs
    

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