import torch

class MLP(torch.nn.Module):
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
    

if __name__ == "__main__":

    layers = [5,3,2]

    model = MLP(layers)
    print(model.layer_sizes)