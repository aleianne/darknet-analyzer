import numpy as np
import matplotlib.pyplot as plt


def create_pyplot_histogram(data, hist_title, x_label):

    if not isinstance(hist_title, str) or not isinstance(x_label, str):
        print("histogram title or x label is not a string")
        return

    # define the number of bins to be used
    num_bins = 100

    n, bins, patches = plt.hist(data, num_bins, normed=1, facecolor='blue', alpha=0.5)

    plt.xlabel(x_label)
    plt.ylabel('frequency')
    plt.title(hist_title)

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.show()
