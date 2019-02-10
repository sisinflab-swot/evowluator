import numpy as np
from typing import Dict, List, Tuple, Union

from matplotlib import pyplot as plt, ticker


def setup_plot(**kwargs) -> (plt.Figure, Union[plt.Axes, List[plt.Axes]]):
    n_figures = kwargs.get('nrows', 0) * kwargs.get('ncols', 0)

    if 'figsize' not in kwargs:
        kwargs['figsize'] = (16, 9) if n_figures > 2 else (10, 9)

    kwargs['squeeze'] = False
    fig, axes = plt.subplots(**kwargs)
    fig.canvas.set_window_title('evOWLuator - results')

    count = 0

    for row in axes:
        for ax in row:
            ax.xaxis.set_major_formatter(default_formatter())
            ax.yaxis.set_major_formatter(default_formatter())
            count += 1

    return fig, axes


def set_scale(ax: plt.Axes, scale: str, axis: str = 'both') -> None:
    """Workaround for formatter getting reset on set_[xy]scale."""
    x_maj = ax.xaxis.get_major_formatter()
    x_min = ax.xaxis.get_minor_formatter()
    y_maj = ax.yaxis.get_major_formatter()
    y_min = ax.yaxis.get_minor_formatter()

    if axis != 'x':
        ax.set_yscale(scale)

    if axis != 'y':
        ax.set_xscale(scale)

    ax.xaxis.set_major_formatter(x_maj)
    ax.xaxis.set_minor_formatter(x_min)
    ax.yaxis.set_major_formatter(y_maj)
    ax.yaxis.set_minor_formatter(y_min)


def draw_min_avg_max_histograms(ax: plt.Axes, data: Dict[str, Tuple[float, float, float]],
                                metric: str, unit: str) -> None:
    reasoners = list(data.keys())
    reasoners.sort()

    n_reasoners = len(reasoners)
    n_stats = 3

    width = 1.0 / (n_reasoners + 1)
    bar_width = 0.8 * width

    for i, reasoner in enumerate(reasoners):
        ax.bar([j + width * i for j in range(n_stats)],
               data[reasoner],
               bar_width, alpha=0.9, label=reasoner)

    display_labels(ax)

    ax.set_title('Minimum, average and maximum {}'.format(metric))

    ylabel = '{} ({})'.format(metric, unit)
    ax.set_ylabel(ylabel[0].upper() + ylabel[1:])

    ax.set_xticks([p + width * ((n_reasoners - 1) / 2) for p in range(n_stats)])
    ax.set_xticklabels(['Min', 'Avg', 'Max'])

    set_scale(ax, 'log', axis='y')
    display_grid(ax, axis='y')
    ax.legend()


def draw_stacked_histograms(ax: plt.Axes, data: Dict[str, List[float]], labels: List[str]) -> None:
    reasoners = list(data.keys())
    reasoners.sort()

    n_sections = len(next(iter(data.values())))
    pos = np.arange(len(reasoners))
    width = 0.35

    values = [data[r][0] for r in reasoners]
    ax.bar(pos, values, width, alpha=0.9, label=labels[0])

    for section in range(1, n_sections):
        prev_values = values
        values = [data[r][section] for r in reasoners]
        ax.bar(pos, values, width, alpha=0.9, bottom=prev_values, label=labels[section])

    display_labels(ax, center=True, fmt='{:.2f}')

    ax.set_xticks(pos)
    ax.set_xticklabels(reasoners)

    display_grid(ax, axis='y')
    ax.legend()


def draw_scatter_plot(ax: plt.Axes, data: Dict[str, Tuple[List[float], List[float]]]) -> None:
    reasoners = list(data.keys())
    reasoners.sort()

    dataset_size = len(next(iter(data.values())))
    point_size = configure_scatter_plot(ax, dataset_size)

    for reasoner in reasoners:
        x, y = data[reasoner]
        ax.scatter(x, y, s=point_size, alpha=0.5, label=reasoner)
        weights = list(range(len(x), 0, -1))
        ax.plot(x, np.poly1d(np.polyfit(x, y, 1, w=weights))(x))

    display_grid(ax)
    ax.legend()


def default_formatter() -> ticker.Formatter:
    return ticker.FormatStrFormatter('%g')


def configure_scatter_plot(ax: plt.Axes, dataset_size: int) -> float:
    if dataset_size > 100:
        set_scale(ax, 'log')
        point_size = 10.0
    else:
        point_size = 50.0

    return point_size


def display_labels(ax: plt.Axes, center: bool = False, fmt: str = '{:.0f}') -> None:
    x_mult = 0.5

    for rect in ax.patches:
        h = rect.get_height()
        w = rect.get_width()

        if center and h > 40.0:
            y_mult = 0.5
            y_add = -15.0
        else:
            y_mult = 1.05
            y_add = 0.0

        x = rect.get_x() + w * x_mult
        y = rect.get_y() + h * y_mult + y_add

        ax.text(x, y, fmt.format(h), ha='center', va='bottom')


def display_grid(ax: plt.Axes, axis: str = 'both') -> None:
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.grid(b=True, axis=axis, which='major')
    ax.grid(b=True, axis=axis, which='minor', alpha=0.25)
