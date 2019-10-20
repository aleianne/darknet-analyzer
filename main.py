from struct import error
from traceback import print_exc

from cfg_file_handler import load_configuration_file
from utils.exceptions import ObjectTypeException
from weights_file import AnalyzeDarknetWeights


def debug_configuration_file_object(cfg):

    print("the number of sections of the configuration file is", len(cfg.get_all_sections()))

    for section in cfg.get_all_sections():
        section.print_all_options()


def analyze_weights(args_params):
    cfg_file_filename = args_params.configuration_file
    weights_file_filename = args_params.weights_file

    try:
        # load the configuration data structure
        cfg = load_configuration_file(cfg_file_filename)
        # debug_configuration_file_object(cfg)

        # load the weights file
        weights_analyzer = AnalyzeDarknetWeights(weights_file_filename, cfg)
        weights_analyzer.analyze_weights()
        weights_analyzer.print_analysis_results()
        weights_analyzer.plot_weight_hist()

    except EnvironmentError as e:
        print("\nException traceback:")
        print_exc()
        print("\nException:", e)
    except ObjectTypeException as e:
        print("\nException traceback:")
        print_exc()
        print("\nException:", e)
    except error as e:
        print("\nException traceback:")
        print_exc()
        print("\nException:", e)


if __name__ == "__main__":
    import argparse as ap

    # parse the argument passed to this script
    parser = ap.ArgumentParser(description="this script analyze the configuration "
                                           "file for a given architecture in DarkNet")
    parser.add_argument("configuration_file")
    parser.add_argument("weights_file")

    args = parser.parse_args()

    analyze_weights(args)
