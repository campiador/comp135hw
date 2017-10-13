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
    plt.show(block=False)
    plt.savefig('part1_accuracies.png')



def plot_accuracies_with_stderr(l_title, l_axis_x, l_axis_y, l1, x1, y1, y1err, l2, x2, y2, y2err,
                                l3, x3, y3, y3err, l4, x4, y4, y4err):

    plt.errorbar(x1,  y1, yerr=y1err, color='r', label=l1, capsize=5)
    plt.plot(x1, y1, 'ro')

    plt.errorbar(x2,  y2, yerr=y2err, color='b', label=l2, capsize=5)
    plt.plot(x2, y2, 'bo')

    plt.errorbar(x3,  y3, yerr=y3err, color='g', label=l3, capsize=5)
    plt.plot(x3, y3, 'gs')

    plt.errorbar(x4,  y4, yerr=y4err, color='y', label=l4, capsize=5)
    plt.plot(x4, y4, 'ys')

    plt.title(l_title)
    plt.xlabel(l_axis_x)
    plt.ylabel(l_axis_y)

    plt.axis([0, 550, 0, 100])
    plt.legend(loc='best')
    plt.show(block=False)
    plt.savefig('./part_2_accuracies_with_stderr.png')


#plotables
#l1_list, x1_list, y1_list, y1err_list


def plot_accuracies_with_stderr_1(l_title, l_axis_x, l_axis_y, l1, x1, y1, y1err):



    plt.errorbar(x1,  y1, yerr=y1err, color='r', label=l1, capsize=5)
    plt.plot(x1, y1, 'ro')

    plt.title(l_title)
    plt.xlabel(l_axis_x)
    plt.ylabel(l_axis_y)

    plt.axis([0, 1000, 0, 1])
    plt.legend(loc='best')
    plt.show(block=True)
    plt.savefig('./hw1_part.png')