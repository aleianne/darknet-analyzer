import struct
import collections
import time

import numpy as np

from utils.util_functions import generate_filename
from utils.constants import CONVOLUTION, CONNECTED, DROPOUT, MAXPOOL
from batch_param_loader import BatchParamLoader
from network_params import NetworkParams, NetworkLayerConvParams, NetworkLayerFcParams, NetworkDropoutLayer, NetworkMaxPoolLayer
from pathlib import Path

class LoadDarknetWeightsFile:
    """
        This class is in charge to read the configuration files for a specified neural network model, in particular reads the
        file that contains the architecture of a specific network and and the weights of the neurons
    """

    def __init__(self, filename, cfg_data):
        self.filename = filename
        self.cfg_data = cfg_data

        self.network_params = None

        self._last_layer = None

        self._nn_input_width = 0
        self._nn_input_height = 0
        self._nn_input_channels = 0

    def load_network_params(self):
        # create the weights file path
        weights_file_path = generate_filename(self.filename)

        # only for debug
        if not weights_file_path.exists():
            print("the weights file", weights_file_path.as_posix(), "doesn't exists")

        # create the network parameters
        self.network_params = NetworkParams()

        with open(weights_file_path, "rb") as file_ptr:

            # decode the first parameter into the weight file
            (major, minor, revision) = self._load_version_info(file_ptr)

            layer_n = 0

            for section in self.cfg_data.get_all_sections():

                # if self._last_layer is None:
                #     print("during the execution of the section", layer_n, "the last layer is None")

                # check the section name
                section_name = section.get_section_name()

                if section_name == "net":

                    self.load_nn_input_dimension(section)

                elif section_name == CONVOLUTION:

                    self._last_layer = self.create_convolution_params_layer(section, layer_n - 1, file_ptr)
                    self.network_params.add_layer_params(self._last_layer)

                elif section_name == CONNECTED:

                    self._last_layer = self.create_connected_params_layer(section, layer_n - 1, file_ptr)
                    self.network_params.add_layer_params(self._last_layer)

                elif section_name == MAXPOOL:

                    self._last_layer = self.create_maxpool_layer(section, layer_n - 1)
                    self.network_params.add_layer_params(self._last_layer)

                elif section_name == DROPOUT:

                    pass

                else:
                    pass

                layer_n += 1

        return self.network_params

    def get_network_params(self):
        return self.network_params

    def load_nn_input_dimension(self, section):

        self._nn_input_height = int(section.find_opt("height"))
        self._nn_input_width = int(section.find_opt("width"))
        self._nn_input_channels = int(section.find_opt("channels"))

    def create_convolution_params_layer(self, section, layer_n, file):
        size = int(section.find_opt("size"))
        filters = int(section.find_opt("filters"))
        pad = int(section.find_opt("pad"))
        stride = int(section.find_opt("stride"))
        batch_norm = section.find_opt("batch_normalize")

        if batch_norm is None:
            batch_norm = 0
        else:
            batch_norm = int(batch_norm)

        input_width = 0
        input_height = 0
        input_channels = 0

        if layer_n == 0:
            input_width = self._nn_input_width
            input_height = self._nn_input_height
            input_channels = self._nn_input_channels
        else:
            input_width = self._last_layer.get_output_width()
            input_height = self._last_layer.get_output_height()
            input_channels = self._last_layer.get_output_channels()

        batch_param_loader = BatchParamLoader(file, filters)

        if batch_norm == 1:
            # load the batch normalization params
            batch_param_loader.load_params()

        conv_layer_params = NetworkLayerConvParams(layer_n, size, pad, filters, stride, input_channels, input_width, input_height)

        conv_layer_params.read_weights_file(file)
        # conv_layer_params.debug()
        return conv_layer_params

    def create_connected_params_layer(self, section, layer_n, file):
        n_output = int(section.find_opt("output"))
        n_input = 0

        if layer_n == 0:
            # todo ora il caso in cui il layer 0 sia un layer fully connected non è preso in considerazione
            pass
        else:

            last_layer_type = self._last_layer.get_layer_type()

            if last_layer_type == CONNECTED:
                n_input = self._last_layer.get_n_output()
            elif last_layer_type == CONVOLUTION:
                n_input = self._last_layer.get_output_neurons()
            elif last_layer_type == MAXPOOL:
                n_input = self._last_layer.get_output_neurons()

        fc_layer_params = NetworkLayerFcParams(layer_n, n_input, n_output)
        fc_layer_params.read_weights_file(file)
        # fc_layer_params.debug()
        return fc_layer_params

    # def _get_layer_params(self, i):
    #
    #     if i == 0:
    #         pass
    #     else:
    #
    #         if self._last_layer.get_layer_type() == CONNECTED:
    #             pass
    #         elif self._last_layer.get_layer_type() == CONVOLUTION:
    #             pass

    def create_maxpool_layer(self, section, layer_n):
        size = int(section.find_opt("size"))
        pad = section.find_opt("pad")
        stride = int(section.find_opt("stride"))

        if pad is None:
            pad = 0
        else:
            pad = int(pad)

        input_width = self._last_layer.get_output_width()
        input_height = self._last_layer.get_output_height()
        input_channels = self._last_layer.get_output_channels()

        maxpool_layer = NetworkMaxPoolLayer(layer_n, stride, pad, size, input_width, input_height, input_channels)
        return maxpool_layer

    def create_dropout_layer(self):
        pass

    def _load_version_info(self, file_ptr):
        bytes_to_int_parser = struct.Struct("i")

        b_major = file_ptr.read(4)
        b_minor = file_ptr.read(4)
        b_revision = file_ptr.read(4)

        major = bytes_to_int_parser.unpack(b_major)[0]
        minor = bytes_to_int_parser.unpack(b_minor)[0]
        revision = bytes_to_int_parser.unpack(b_revision)[0]

        # only for debugging
        # print("\n*** PRINT DARKNET VERSION INFO ***")
        # print("the version is {M}.{m}.{r}\n".format(M=major, m=minor, r=revision))

        return major, minor, revision
    #
    # def _load_weights_and_biases(self, n_weights, n_biases, file):
    #     # first of all load the biases stored into the weight file
    #     biases = []
    #     self._decode_bytes_from_file(file, n_biases, biases)
    #
    #     # load the weights associated to a specific layer
    #     weights = []
    #     self._decode_bytes_from_file(file, n_weights, weights)
    #
    #     return biases, weights
    #
    # def _load_connected_layer(self, n_input, n_output, file):
    #     weights_n = n_input * n_output
    #
    #
    #     biases = []
    #     self._decode_bytes_from_file(file, n_output, biases)
    #
    #     weights = []
    #     self._decode_bytes_from_file(file, weights_n, weights)
    #
    #     return biases, weights


class AnalyzeDarknetWeights:

    def __init__(self, filename, cfg_data):

        self.filename = Path(filename)

        # load the networks param from the file
        print("Begin to load the parameters from the {filename} file...".format(filename=self.filename.name))
        start = time.time()

        weights_loader = LoadDarknetWeightsFile(filename, cfg_data)
        self.network_params = weights_loader.load_network_params()

        interval = round(time.time() - start, 2)
        print("parameters read correctly in {sec} seconds!".format(sec=interval))

        self.g_weights_acc = 0          # weight accumulator variable used to store the sum of all the weight
        self.g_biases_acc = 0           # biases accumulator variable used to store the sum of all the biases
        self.g_weights_num = 0
        self.g_biases_num = 0

        self.w_min = 0
        self.w_max = 0
        self.b_min = 0
        self.b_max = 0

        self.layer_avg_list = []

    def analyze_weights(self):

        self.g_weights_acc = 0
        self.g_biases_acc = 0
        self.g_biases_num = 0
        self.g_weights_num = 0

        self.w_min = 0
        self.w_max = 0
        self.b_min = 0
        self.b_max = 0

        self.layer_avg_list = []

        print("\nBegin the weights analysis...")
        start = time.time()
        self.analyze_layers()
        interval = round(time.time() - start, 2)
        print("analysis analysis completed in {sec} seconds!".format(sec=interval))

    def analyze_layers(self):

        # create the data average namedtuple
        DataAverage = collections.namedtuple("DataAverage", ["biases_avg", "weights_avg", "b_min", "b_max", "w_min", "w_max"])

        layers = self.network_params.get_layers()
        for layer in layers:
            if layer.get_layer_type() == CONVOLUTION or layer.get_layer_type() == CONNECTED:
                biases_avg, weights_avg, b_min_value, b_max_value, w_min_value, w_max_value = self.analyze_single_layer(layer)
                d = DataAverage(biases_avg, weights_avg, b_min_value, b_max_value, w_min_value, w_min_value)
                self.layer_avg_list.append(d)

        self._compute_global_max_min()

    def analyze_single_layer(self, layer):
        biases = layer.get_biases()
        weights = layer.get_weights()

        layer_b_acc = 0
        layer_w_acc = 0

        for bias in biases:
            layer_b_acc += bias
            self.g_biases_acc += bias

        for weight in weights:
            layer_w_acc += weight
            self.g_weights_acc += weight

        self.g_biases_num += biases.size
        self.g_weights_num += weights.size

        layer_b_avg = layer_b_acc / biases.size
        layer_w_avg = layer_w_acc / weights.size
        layer_b_min_value = np.amin(biases)
        layer_b_max_value = np.amax(biases)
        layer_w_min_value = np.amin(weights)
        layer_w_max_value = np.amax(weights)

        return layer_b_avg, layer_w_avg, layer_b_min_value, layer_b_max_value, layer_w_min_value, layer_w_max_value

    def print_values(self):
        print("\n--- {filename} analysis results:".format(filename=self.filename.name))
        print("the total number of weights into the architecture is", self.g_weights_num)
        print("the total number of biases into the architecture is", self.g_biases_num)
        print("the value of weights are between {min} and {max}".format(min=self.w_min, max=self.w_max))
        print("the average between all the weights is", self.g_weights_acc / self.g_weights_num)
        print("the value of biases are between {min} and {max}".format(min=self.b_min, max=self.b_max))
        print("the average between all the weighs is", self.g_biases_acc / self.g_biases_num)

    def _compute_global_max_min(self):

        max_b = -100
        min_b = 100
        max_w = -100
        min_w = 100

        for d in self.layer_avg_list:

            if max_b < d.b_max:
                max_b = d.b_max

            if min_b > d.b_min:
                min_b = d.b_min

            if max_w < d.w_max:
                max_w = d.w_max

            if min_w > d.w_min:
                min_w = d.w_min

        self.b_min = min_b
        self.b_max = max_b
        self.w_min = min_w
        self.w_max = max_w

    def _debug_network_params(self, layer_n):

        if layer_n < 0 or layer_n >= len(self.network_params.get_layers()):
            print("impossible to debug the layer {layer} because is less than zero or excede the list size".format(layer=layer_n))
            return

        layers = self.network_params.get_layers()
        layer = layers[layer_n]

        if not (layer.get_layer_type() == CONVOLUTION or layer.get_layer_type() == CONNECTED):
            print("impossible to show the weights of the {layer} layer".format(layer=layer.get_layer_type()))
            return

        biases_n = layer.get_weights().size
        weights_n = layer.get_biases().size

        print("the number of biases is {n}".format(n=biases_n))
        print("the biases are:")
        for bias in layer.get_biases():
            print(bias, end=", ")

        print("\n\nthe number of weights is {n}".format(n=weights_n))
        print("the weights are:")
        for weight in layer.get_weights():
            print(weight, end=", ")
