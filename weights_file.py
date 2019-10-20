import collections
import struct
import time
from pathlib import Path

import numpy as np

from batch_param_loader import BatchParamLoader
from matplot_histograms import create_pyplot_histogram
from network_params import NetworkParams, NetworkLayerConvParams, \
    NetworkLayerFcParams, NetworkMaxPoolLayer
from utils.constants import CONVOLUTION, CONNECTED, DROPOUT, MAXPOOL
from utils.util_functions import generate_filename


class LoadDarknetWeightsFile:
    """
        This class is in charge to read the configuration files for a specified neural network model, in particular reads the
        file that contains the architecture of a specific network and and the weights of the neurons
    """

    def __init__(self, weights_filename, cfg_data):
        self.weights_filename = weights_filename
        self.cfg_data = cfg_data

        self.network_params = None

        self._last_layer = None

        self._nn_input_width = 0
        self._nn_input_height = 0
        self._nn_input_channels = 0

    def load_network_params(self):
        # create the weights file path
        weights_file_path = self.weights_filename

        # only for debug
        if not weights_file_path.exists():
            print("the weights file", weights_file_path.as_posix(), "doesn't exists")
            raise FileNotFoundError

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

        conv_layer_params = NetworkLayerConvParams(layer_n, size, pad, filters, stride, input_channels, input_width,
                                                   input_height)

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

    def __init__(self, weights_filename, cfg_object):

        self.weights_filename = Path(weights_filename)

        # load network params from file
        print("Begin to load parameters from {filename}...".format(filename=self.weights_filename.name))
        start = time.time()

        weights_loader = LoadDarknetWeightsFile(self.weights_filename, cfg_object)
        self.network_params = weights_loader.load_network_params()

        interval = round(time.time() - start, 2)
        print("parameters read correctly in {sec} seconds!".format(sec=interval))

        self.g_weights_acc = 0  # weight accumulator variable used to store the sum of all the weight
        self.g_biases_acc = 0  # biases accumulator variable used to store the sum of all the biases
        self.g_weights_num = 0  # variable that store the number of weights
        self.g_biases_num = 0  # variable that store the number of biases

        self.w_min = 0
        self.w_max = 0
        self.b_min = 0
        self.b_max = 0

        self.layer_avg_list = []

    def plot_weight_hist(self):

        total_weights = np.array([])

        for layer in self.network_params.get_layers():
            l_type = layer.get_layer_type()
            if l_type == CONVOLUTION or l_type == CONNECTED:
                total_weights = np.append(total_weights, layer.get_weights())

        print("total number of weights stored into the system {n}".format(n=total_weights.size))
        print("begin to plot the weights histogram")

        create_pyplot_histogram(total_weights, "weight distribution", "frequency")

    def analyze_weights(self):

        # before to start a new analysis clean all the instance variable
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
        print("analysis completed in {sec} seconds!".format(sec=interval))

    def analyze_layers(self):

        # create the data average namedtuple
        DataAverage = collections.namedtuple("DataAverage",
                                             ["layer_n", "layer_name", "biases_avg", "weights_avg", "b_min", "b_max",
                                              "w_min", "w_max"])

        layers = self.network_params.get_layers()
        for layer in layers:

            n = layer.get_layer_n()
            l_type = layer.get_layer_type()

            if l_type == CONVOLUTION or l_type == CONNECTED:
                biases_avg, weights_avg, b_min_value, b_max_value, w_min_value, w_max_value = self.analyze_single_layer(layer)
                d = DataAverage(n, l_type, biases_avg, weights_avg, b_min_value, b_max_value, w_min_value, w_max_value)
                self.layer_avg_list.append(d)

        self._compute_global_max_min()

    def analyze_single_layer(self, layer):
        biases = layer.get_biases()
        weights = layer.get_weights()

        layer_b_acc = 0
        layer_w_acc = 0

        for bias in biases:
            layer_b_acc += abs(bias)
            self.g_biases_acc += abs(bias)

        for weight in weights:
            layer_w_acc += abs(weight)
            self.g_weights_acc += abs(weight)

        self.g_biases_num += biases.size
        self.g_weights_num += weights.size

        layer_b_avg = layer_b_acc / biases.size
        layer_w_avg = layer_w_acc / weights.size
        layer_b_min_value = np.amin(biases)
        layer_b_max_value = np.amax(biases)
        layer_w_min_value = np.amin(weights)
        layer_w_max_value = np.amax(weights)

        return layer_b_avg, layer_w_avg, layer_b_min_value, layer_b_max_value, layer_w_min_value, layer_w_max_value

    def print_analysis_results(self):
        i = 1
        print("\n--- {filename} analysis results:".format(filename=self.weights_filename.name))
        print("the total number of weights into the architecture is", self.g_weights_num)
        print("the total number of biases into the architecture is", self.g_biases_num)
        print("the value of weights are between {min} and {max}".format(min=self.w_min, max=self.w_max))
        print("the average between all the weights is", self.g_weights_acc / self.g_weights_num)
        print("the value of biases are between {min} and {max}".format(min=self.b_min, max=self.b_max))
        print("the average between all the biases is", self.g_biases_acc / self.g_biases_num)
        print("\n-- single layer analysis")
        for param in self.layer_avg_list:
            print("Layer {n} data:".format(n=i))
            print("the value of weights are between {min} and {max}".format(min=param.w_min, max=param.w_max))
            print("the average between all the weights is", param.weights_avg)
            print("the value of biases are between {min} and {max}".format(min=param.b_min, max=param.b_max))
            print("the average between all the biases is", param.biases_avg)
            i += 1

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
