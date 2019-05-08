import numpy as np

from utils.constants import CONVOLUTION, CONNECTED, DROPOUT, MAXPOOL
from utils.exceptions import LayerTypeException
from utils.util_functions import decode_bytes_from_file


class BaseNetworkLayerParams:

    def __init__(self, layer_type, layer_n):
        self.type = layer_type
        self.layer_n = layer_n

    def get_layer_type(self):
        return self.type

    def get_layer_n(self):
        return self.layer_n


class NetworkLayerParams(BaseNetworkLayerParams):
    """
        This class store all the parameters that a network model need
    """

    def __init__(self, layer_type, layer_n, weights=None, biases=None):

        super().__init__(layer_type, layer_n)

        if weights is None:
            weights = []

        if biases is None:
            biases = []

        self.weights = np.array(weights, dtype=float)
        self.biases = np.array(biases, dtype=float)

        self.type = layer_type
        self.layer_n = layer_n

    def get_weights(self):
        return self.weights

    def get_biases(self):
        return self.biases

    def append_biases(self, biases):
        self.biases = np.append(self.biases, biases)

    def append_weights(self, weights):
        self.weights = np.append(self.weights, weights)

    def load_weights_and_biases_from_file(self, b, w, file):
        # first of all load the biases stored into the weight file
        biases = []
        decode_bytes_from_file(file, b, biases)
        self.append_biases(biases)

        # load the weights associated to a specific layer
        weights = []
        decode_bytes_from_file(file, w, weights)
        self.append_weights(weights)

        return biases, weights


class NetworkLayerConvParams(NetworkLayerParams):

    def __init__(self, layer_n, size, pad, filters, stride, input_channels, input_width, input_height):
        super().__init__(CONVOLUTION, layer_n)

        self.filters = filters
        self.size = size
        self.stride = stride

        if pad == 1:
            self.pad = int(self.size / 2)
        else:
            self.pad = 0

        self.input_channels = input_channels
        self.input_width = input_width
        self.input_height = input_height

        self.output_height = int((self.input_height - self.size + 2 * self.pad) / self.stride + 1)
        self.output_width = int((self.input_width - self.size + 2 * self.pad) / self.stride + 1)
        self.output_neurons = int(self.output_height * self.output_width * self.filters)

        self.n_weights = (self.size ** 2) * self.input_channels * self.filters

    def get_filters(self):
        return self.filters

    def get_input_channels(self):
        return self.input_channels

    def get_input_width(self):
        return self.input_width

    def get_input_height(self):
        return self.input_height

    def get_n_weights(self):
        return self.n_weights

    def get_output_neurons(self):
        return self.output_neurons

    def get_output_channels(self):
        return self.filters

    def get_output_width(self):
        return self.output_width

    def get_output_height(self):
        return self.output_height

    def debug(self):
        print("\n*** DEBUG ***")
        print("layer {n}, name {name}".format(n=self.layer_n, name=self.type))

        print("the number of filters is", self.filters)
        print("the filter size is", self.size)
        print("the stride is", self.stride)
        print("the pad is", self.pad)
        print("the input channels are", self.input_channels)
        print("the input width is", self.input_width)
        print("the input height is", self.input_height)
        print("the output neurons are", self.output_neurons)
        print("the output width is", self.output_width)
        print("the output height is", self.output_height)
        print("the number of weights is", self.n_weights)

        print("\n*** END DEBUG ***")

    def read_weights_file(self, file):
        self.load_weights_and_biases_from_file(self.filters, self.n_weights, file)


class NetworkLayerFcParams(NetworkLayerParams):

    def __init__(self, layer_n, n_input, n_output):
        super().__init__(CONNECTED, layer_n)

        self.n_input = n_input
        self.n_output = n_output

        self.n_weights = self.n_input * self.n_output

    def get_n_weights(self):
        return self.n_weights

    def get_n_input(self):
        return self.n_input

    def get_n_output(self):
        return self.n_output

    def debug(self):
        print("\n*** DEBUG ***")
        print("the number of inputs is", self.n_input)
        print("the number of outputs is", self.n_output)
        print("the number of weight is", self.n_weights)
        print("\n")

    def read_weights_file(self, file):
        self.load_weights_and_biases_from_file(self.n_output, self.n_weights, file)


class NetworkDropoutLayer(BaseNetworkLayerParams):

    # TODO per ora nel layer dropout non aggiungere niente altro

    def __init__(self, layer_n):
        super.__init__(DROPOUT, layer_n)


class NetworkMaxPoolLayer(BaseNetworkLayerParams):

    def __init__(self, layer_n, stride, pad, size, input_width, input_height, input_channels):
        super().__init__(MAXPOOL, layer_n)

        self.stride = stride
        self.size = size

        if pad == 0:
            self.pad = self.size - 1
        else:
            self.pad = pad

        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels

        self.output_width = int((self.input_width + self.pad - self.size) / self.stride + 1)
        self.output_height = int((self.input_height + self.pad - self.size) / self.stride + 1)
        self.output_channels = self.input_channels

        self.output_neurons = self.output_height * self.output_width * self.output_channels

    def get_output_channels(self):
        return self.output_channels

    def get_output_width(self):
        return self.output_width

    def get_output_height(self):
        return self.output_height

    def get_input_channels(self):
        return self.input_channels

    def get_input_width(self):
        return self.input_width

    def get_input_height(self):
        return self.input_height

    def get_output_neurons(self):
        return self.output_neurons


class NetworkParams:

    def __init__(self):
        self.layers = []

    def add_layer_params(self, layer):
        if isinstance(layer, BaseNetworkLayerParams):
            self.layers.append(layer)
        else:
            raise LayerTypeException

    def get_layers(self):
        return self.layers

    def get_last_layer(self):
        return self.layers[-1]
