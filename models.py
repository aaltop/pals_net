import abc

from pytorch_helpers import pretty_print

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
    

if __name__ == "__main__":

    layers = [5,3,2]

    model = MLP(layers)
    print(model.layer_sizes)
    print(model)
    model2 = MLP(**model.instantiation_kwargs)
    print(model2)