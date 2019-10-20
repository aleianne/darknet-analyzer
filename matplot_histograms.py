import matplotlib.pyplot as plt

from matplotlib import colors

def create_pyplot_histogram(data, hist_title, x_label):

    if not isinstance(hist_title, str) or not isinstance(x_label, str):
        print("histogram title or x label is not a string")
        return

    # define the number of bins to be used
    num_bins = 75

    n, bins, patches = plt.hist(data, num_bins)

    plt.xlabel('weights value')
    plt.ylabel('frequency %')
    plt.title('Weights histogram')
    plt.yscale(value='log', nonposy='clip')

    # # We'll color code by height, but you could use any scalar
    # fracs = n / n.max()
    #
    # # we need to normalize the data to 0..1 for the full range of the colormap
    # norm = colors.Normalize(fracs.min(), fracs.max())
    #
    # # Now, we'll loop through our objects and set the color of each accordingly
    # for thisfrac, thispatch in zip(fracs, patches):
    #     color = plt.cm.viridis(norm(thisfrac))
    #     thispatch.set_facecolor(color)

    plt.subplots_adjust(left=0.15)
    #plt.show()
    plt.savefig('./images/weights_hist.png')
