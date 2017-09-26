import matplotlib.pyplot as plt


def plot_accuracies(l_title, l_axis_x, l_axis_y, l1, x1, y1, l2, x2, y2):
    plt.plot(x1, y1, color='r', label=l1)
    plt.plot(x1, y1, 'ro')

    plt.plot(x2, y2, color='b', label=l2)
    plt.plot(x2, y2, 'bs')

    plt.title(l_title)
    plt.xlabel(l_axis_x)
    plt.ylabel(l_axis_y)

    plt.axis([0, 100, 0, 100])
    plt.legend(loc='best')
    plt.show()

