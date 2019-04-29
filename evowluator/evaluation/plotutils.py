import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from matplotlib import pyplot as plt, ticker


def setup_plot(**kwargs) -> (plt.Figure, Union[plt.Axes, List[plt.Axes]]):
    n_figures = kwargs.get('nrows', 0) * kwargs.get('ncols', 0)

    if 'figsize' not in kwargs:
        height = 5 if n_figures == 1 else 9
        width = 10 if n_figures <= 2 else 16
        kwargs['figsize'] = (width, height)

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


def draw_histograms(ax: plt.Axes, data: Dict[str, float], metric: str,
                    unit: Optional[str] = None) -> None:
    reasoners = list(data.keys())
    reasoners.sort()
    n_reasoners = len(reasoners)

    width = (1.0 / n_reasoners) * 0.8

    for i, reasoner in enumerate(reasoners):
        ax.bar([i], data[reasoner], width=width, alpha=0.9, label=reasoner)

    display_labels(ax)

    ax.set_title(metric[0].upper() + metric[1:])

    ax.set_xticks(np.arange(n_reasoners))
    ax.set_xticklabels(reasoners)

    ylabel = '{} ({})'.format(metric, unit) if unit else metric
    ax.set_ylabel(ylabel[0].upper() + ylabel[1:])

    display_grid(ax, axis='y')
    legend = ax.legend()
    legend.set_draggable(True)


def draw_grouped_histograms(ax: plt.Axes, data: Dict[str, List[float]], metrics: List[str]) -> None:
    configure_histogram_plot(ax, data)

    reasoners = list(data.keys())
    reasoners.sort()

    n_reasoners = len(reasoners)
    n_stats = len(metrics)

    width = 1.0 / (n_reasoners + 1)
    bar_width = 0.8 * width

    for i, reasoner in enumerate(reasoners):
        ax.bar([j + width * i for j in range(n_stats)], data[reasoner],
               width=bar_width, alpha=0.9, label=reasoner)

    display_labels(ax)

    ax.set_xticks([p + width * ((n_reasoners - 1) / 2) for p in range(n_stats)])
    ax.set_xticklabels(metrics)

    display_grid(ax, axis='y')

    legend = ax.legend()
    legend.set_draggable(True)


def draw_min_avg_max_histograms(ax: plt.Axes, data: Dict[str, List[float]],
                                metric: str, unit: str) -> None:
    ax.set_title('Minimum, average and maximum {}'.format(metric))

    ylabel = '{} ({})'.format(metric, unit)
    ax.set_ylabel(ylabel[0].upper() + ylabel[1:])

    draw_grouped_histograms(ax, data, ['Min', 'Avg', 'Max'])


def draw_stacked_histograms(ax: plt.Axes, data: Dict[str, List[float]], labels: List[str]) -> None:
    configure_histogram_plot(ax, data, stacked=True)

    reasoners = list(data.keys())
    reasoners.sort()
    n_reasoners = len(reasoners)

    n_sections = len(next(iter(data.values())))
    pos = np.arange(n_reasoners)
    width = 0.5

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

    legend = ax.legend()
    legend.set_draggable(True)


def configure_histogram_plot(ax: plt.Axes, data: Dict[str, List[float]],
                             stacked: bool = False) -> None:
    if stacked:
        data_min = min(p for l in data.values() for p in l)
        data_max = max(sum(l) for l in data.values())
    else:
        data_min = min(p for l in data.values() for p in l)
        data_max = max(p for l in data.values() for p in l)

    if data_max / data_min > 50.0:
        set_scale(ax, 'symlog', axis='y')
        data_min = 10.0 ** np.floor(np.log10(data_min))
        data_max = 10.0 ** np.ceil(np.log10(data_max))
        ax.set_ylim(bottom=data_min, top=data_max)


def draw_scatter_plot(ax: plt.Axes, data: Dict[str, Tuple[List[float], List[float]]]) -> None:
    reasoners = list(data.keys())
    reasoners.sort()

    dataset_size = len(next(iter(data.values()))[0])
    point_size = configure_scatter_plot(ax, dataset_size)

    for reasoner in reasoners:
        x, y = data[reasoner]
        ax.scatter(x, y, s=point_size, alpha=0.5, label=reasoner)
        draw_polyline(ax, x, y)

    display_grid(ax)

    legend = ax.legend()
    legend.set_draggable(True)


def draw_polyline(ax: plt.Axes, x: List[float], y: List[float]) -> None:
    data_point_count = len(x)
    weights = [1.0] * data_point_count

    # Force start from first data points
    data_point_count = max(data_point_count // 100, 1)
    y[0] = sum(y[:data_point_count]) / data_point_count
    weights[0] = max(y) * 10.0

    ax.plot(x, np.poly1d(np.polyfit(x, y, 1, w=weights))(x))


def configure_scatter_plot(ax: plt.Axes, dataset_size: int) -> float:
    if dataset_size > 100:
        set_scale(ax, 'log')
        point_size = 10.0
    else:
        point_size = 50.0

    return point_size


def default_formatter() -> ticker.Formatter:
    return ticker.FormatStrFormatter('%g')


def display_labels(ax: plt.Axes, center: bool = False, fmt: str = '{:.0f}') -> None:
    fig = ax.get_figure()
    transform = fig.dpi_scale_trans.inverted()

    x_mult = 0.5

    for rect in ax.patches:
        height_px = rect.get_bbox().transformed(transform).height * fig.dpi

        w = rect.get_width()
        h = rect.get_height()

        if center and height_px > 20.0:
            y_mult = 0.5
            va = 'center'
        else:
            y_mult = 1.0
            va = 'bottom'

        x = rect.get_x() + w * x_mult
        y = rect.get_y() + h * y_mult

        annotation = ax.annotate(fmt.format(h), (x, y), ha='center', va=va)
        annotation.draggable()


def display_grid(ax: plt.Axes, axis: str = 'both') -> None:
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.grid(b=True, axis=axis, which='major')
    ax.grid(b=True, axis=axis, which='minor', alpha=0.25)
