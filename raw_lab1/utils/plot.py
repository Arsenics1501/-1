import matplotlib.pyplot as plt
from easypyplot.barchart import draw
from easypyplot.pdf import plot_open

data = [
    [1, 1, 1, 1],
    [1.1, 1.2, 0.9, 0.8],
    [0.8, 0.9, 2.0, 1.1],
    [1.0, 1.1, 1.2, 1.2],
    [0.8, 0.9, 2.0, 1.1],
    [1.0, 1.1, 1.2, 1.2],
]

with plot_open("lab") as fig:
    ax = fig.gca()
    draw(ax, data, breakdown=False,
            group_names=["mnk", "mkn", "kmn", "nmk", "nkm", "knm"],
            entry_names=["1.txt", "2.txt", "3.txt", "4.txt"],
            )
    ax.set_xlabel('Different implementations of compute kernel')
    ax.set_ylabel('Speedup')
    ax.set_title('Task 1')
